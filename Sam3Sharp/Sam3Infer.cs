using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Tokenizers.DotNet;
using Size = SixLabors.ImageSharp.Size;

namespace Sam3Sharp;

public class Sam3Infer
{
	private readonly Sam3InferConfig _config;

	private readonly InferenceSession _vision_encoder;
	private readonly InferenceSession _text_encoder;
	private readonly InferenceSession _decoder;
	private readonly Tokenizer _tokenizer;

	//目前是写死的
	const int input_image_width_ = 1008;
	const int input_image_height_ = 1008;

	public InferenceSession CreateInferSession(string model)
	{
		var modeldata = File.ReadAllBytes(model);
		if (_config.use_cuda)
		{
			var opt = SessionOptions.MakeSessionOptionWithCudaProvider(0);
			opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
			return new InferenceSession(modeldata, opt);
		}

		return new InferenceSession(modeldata);
	}

	public Sam3Infer(Sam3InferConfig config)
	{
		_config = config;
		var modeldir = config.model_path;
		_vision_encoder = CreateInferSession(Path.Combine(modeldir, "vision-encoder-fp16.onnx"));
		_text_encoder = CreateInferSession(Path.Combine(modeldir, "text-encoder-fp16.onnx"));
		//这个暂未使用
		//_geometry_encoder=CreateInferSession(Path.Combine(modeldir, "geometry-encoder-fp16.onnx"));
		_decoder = CreateInferSession(Path.Combine(modeldir, "decoder-fp16.onnx"));
		_tokenizer = new Tokenizer(Path.Combine(modeldir, "tokenizer.json"));
	}

	public Sam3Result Handle(Sam3Session session)
	{
		lock (this)
		{
			EncodeImage(session);
			EncodeText(session);
			Decode(session);
			session.GCInput();
			var result = PostProcessor(session);
			session.GCOutput();
			return result;
		}
	}


	public void Decode(Sam3Session session)
	{
		var omd_fpn_feat_0 = _decoder.InputMetadata["fpn_feat_0"];
		var alt_fpn_feat_0 =
			new DenseTensor<float>(session.fpn_feat_0.Buffer,
				new[] { 1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2], omd_fpn_feat_0.Dimensions[3] });

		var alt_fpn_feat_1 =
			new DenseTensor<float>(session.fpn_feat_1.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 2, omd_fpn_feat_0.Dimensions[3] / 2
				});

		var alt_fpn_feat_2 =
			new DenseTensor<float>(session.fpn_feat_2.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 4, omd_fpn_feat_0.Dimensions[3] / 4
				});

		var alt_fpn_pos_2 =
			new DenseTensor<float>(session.fpn_pos_2.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 4, omd_fpn_feat_0.Dimensions[3] / 4
				});

		var promptlen = _text_encoder.InputMetadata["input_ids"].Dimensions[1];
		var alt_text_features =
			new DenseTensor<float>(session.text_features.Buffer[0..(promptlen * 256)],
				new[] { 1, promptlen, 256 });

		var alt_text_mask =
			new DenseTensor<bool>(session.text_mask.Buffer[0..promptlen],
				new[] { 1, promptlen });

		using var results = _decoder.Run(new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("fpn_feat_0", alt_fpn_feat_0),
			NamedOnnxValue.CreateFromTensor("fpn_feat_1", alt_fpn_feat_1),
			NamedOnnxValue.CreateFromTensor("fpn_feat_2", alt_fpn_feat_2),
			NamedOnnxValue.CreateFromTensor("fpn_pos_2", alt_fpn_pos_2),
			NamedOnnxValue.CreateFromTensor("prompt_features", alt_text_features),
			NamedOnnxValue.CreateFromTensor("prompt_mask", alt_text_mask),
		});
		session.dt_pred_masks = results[0].AsTensor<float>().ToDenseTensor();
		session.dt_pred_boxes = results[1].AsTensor<float>().ToDenseTensor();
		session.dt_pred_logits = results[2].AsTensor<float>().ToDenseTensor();
		session.dt_presence_logits = results[3].AsTensor<float>().ToDenseTensor();
	}


	public static float Sigmoid(double value)
	{
		return 1.0f / (1.0f + (float)Math.Exp(-value));
	}

	public static int LimitRange(int v, int min, int max)
	{
		if (v <= min)
			return min;
		if (v >= max)
			return max - 1;
		return v;
	}

	public Sam3Result PostProcessor(Sam3Session session)
	{
		var result = new Sam3Result();
		result.mask_model_width = 288; //这个应该从decoder的dim拿
		result.mask_model_height = 288; //这个应该从decoder的dim拿

		//画出包围盒
		var presence_score = Sigmoid(session.dt_presence_logits.GetValue(0));
		var boxdata = session.dt_pred_boxes.Buffer;
		var maskdata = session.dt_pred_masks.Buffer;
		for (int i = 0; i < session.dt_pred_logits.Length; i++)
		{
			float score = Sigmoid(session.dt_pred_logits.GetValue(i)) * presence_score;
			if (score <= 0.4f)
				continue;
			var x1 = (int)(boxdata.Span[i * 4 + 0] * 1008);
			var y1 = (int)(boxdata.Span[i * 4 + 1] * 1008);
			var x2 = (int)(boxdata.Span[i * 4 + 2] * 1008);
			var y2 = (int)(boxdata.Span[i * 4 + 3] * 1008);

			x1 = LimitRange(x1, 0, input_image_width_);
			y1 = LimitRange(y1, 0, input_image_height_);
			x2 = LimitRange(x2, 0, input_image_width_);
			y2 = LimitRange(y2, 0, input_image_height_);
			if (x2 <= x1)
				continue;
			if (y2 <= y1)
				continue;
			var rc = new RectangleF(x1, y1, x2 - x1, y2 - y1);

			var startpos = i * 288 * 288;
			var curmask = maskdata.Span.Slice(startpos, 288 * 288);
			var maskdat = curmask.ToArray();
			var item = new Sam3ResultItem()
			{
				score = score,
				box = rc,
				mask = maskdat
			};
			result.items.Add(item);
		}

		return result;
	}

	public void EncodeImage(Sam3Session session)
	{
		var img = session.org_image;
		//现在是扩大到输入的大小，然后从左上贴过去。
		var destpos = CalcResizeInfo(img.Width, img.Height);
		session.emb_image_width = destpos.w;
		session.emb_image_height = destpos.h;
		session.input_image = img.Clone(x =>
		{
			x.Resize(new ResizeOptions
			{
				Mode = ResizeMode.Manual,
				Position = AnchorPositionMode.TopLeft,
				Size = new Size(input_image_width_, input_image_height_),
				Compand = false,
				PadColor = Color.White,
				TargetRectangle = new Rectangle(0, 0, destpos.w, destpos.h)
			});
		});
		var inputTensorValues = new float[input_image_width_ * input_image_height_ * 3];

		var memtensor = new MemoryTensor<float>(inputTensorValues, _vision_encoder.InputMetadata["images"].Dimensions);
		PixelsNormalizer.NormalizerPixelsToTensor(session.input_image, memtensor, new Vector<int>(0, 0));
		var inputTensor =
			new DenseTensor<float>(inputTensorValues, new[] { 1, 3, input_image_width_, input_image_height_ });
		using var results = _vision_encoder.Run(new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("images", inputTensor),
		});

		//看看是不是可以释放掉
		inputTensorValues = null;

		session.fpn_feat_0 = results[0].AsTensor<float>().ToDenseTensor();
		session.fpn_feat_1 = results[1].AsTensor<float>().ToDenseTensor();
		session.fpn_feat_2 = results[2].AsTensor<float>().ToDenseTensor();
		session.fpn_pos_2 = results[3].AsTensor<float>().ToDenseTensor();
	}

	public void EncodeText(Sam3Session session)
	{
		var ids = _tokenizer.Encode(session.prompt_text).ToList();
		if (ids.Count > 32)
			ids = ids.Take(32).ToList();
		var masklen = ids.Count;
		//开始padding
		if (ids.Count < 32)
		{
			var padlen = 32 - ids.Count;
			for (int i = 0; i < padlen; i++)
			{
				ids.Add(49407);
			}
		}

		//开始encoding
		//32个long!!!
		long[] text_input_ids = new long[32 * 8];
		for (int i = 0; i < 32; i++)
			text_input_ids[i] = ids[i];
		long[] text_attention_mask = new long[32 * 8];
		for (int i = 0; i < masklen; i++)
			text_attention_mask[i] = 1;
		for (int i = masklen; i < 32; i++)
			text_attention_mask[i] = 0;
		var text_input_ids_tensor = new DenseTensor<long>(text_input_ids, [8, 32]);
		var text_attention_mask_tensor = new DenseTensor<long>(text_attention_mask, [8, 32]);
		using var results = _text_encoder.Run(new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("input_ids", text_input_ids_tensor),
			NamedOnnxValue.CreateFromTensor("attention_mask", text_attention_mask_tensor),
		});
		session.text_features = results.First().AsTensor<float>().ToDenseTensor();
		session.text_mask = results[1].AsTensor<bool>().ToDenseTensor();
	}

	//计算出缩放后宽高,实际图像为固定,这个是从左上角开始的宽高
	public (int w, int h) CalcResizeInfo(int width, int height)
	{
		var modelimgratio = (decimal)input_image_width_ / (decimal)input_image_height_;
		var curration = (decimal)width / (decimal)height;
		if (curration >= modelimgratio)
		{
			var ratio = (decimal)width / (decimal)input_image_width_;
			var h = (int)(height / ratio);
			if (h > input_image_height_)
				h = input_image_height_;
			return (input_image_width_, h);
		}
		else
		{
			var ratio = (decimal)height / (decimal)input_image_height_;
			var w = (int)(width / ratio);
			if (w > input_image_width_)
				w = input_image_width_;
			return (w, input_image_height_);
		}
	}
}