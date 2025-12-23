using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Tokenizers.DotNet;

namespace Sam3Sharp;

public class Sam3Infer
{
	private Sam3InferConfig _config;

	private InferenceSession _vision_encoder;
	private InferenceSession _text_encoder;
	private InferenceSession _decoder;
	private readonly Tokenizer _tokenizer;
	private static DataConverter _dataConverter = DataConverter.Native;

	int input_image_width_ = 1008;
	int input_image_height_ = 1008;

	private (int width, int height) original_image_sizes_ = (0, 0);

	public Sam3Infer(Sam3InferConfig config)
	{
		_config = config;
		_vision_encoder = new InferenceSession(config.vision_encoder_path);
		_text_encoder = new InferenceSession(config.text_encoder_path);
		_decoder = new InferenceSession(config.decoder_path);
		_tokenizer = new Tokenizer(config.tokenizer_path);
	}

	private DenseTensor<float> text_features;
	private DenseTensor<bool> text_mask;

	private DenseTensor<float> fpn_feat_0;
	private DenseTensor<float> fpn_feat_1;
	private DenseTensor<float> fpn_feat_2;
	private DenseTensor<float> fpn_pos_2;

	public void Gc()
	{
		text_features = null;
		text_mask = null;
		fpn_feat_0 = null;
		fpn_feat_1 = null;
		fpn_feat_2 = null;
		fpn_pos_2 = null;
		GC.Collect();
	}

	public void Decode()
	{
		var omd_fpn_feat_0 = _decoder.InputMetadata["fpn_feat_0"];
		var alt_fpn_feat_0 =
			new DenseTensor<float>(fpn_feat_0.Buffer,
				new[] { 1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2], omd_fpn_feat_0.Dimensions[3] });

		var alt_fpn_feat_1 =
			new DenseTensor<float>(fpn_feat_1.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 2, omd_fpn_feat_0.Dimensions[3] / 2
				});

		var alt_fpn_feat_2 =
			new DenseTensor<float>(fpn_feat_2.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 4, omd_fpn_feat_0.Dimensions[3] / 4
				});

		var alt_fpn_pos_2 =
			new DenseTensor<float>(fpn_pos_2.Buffer,
				new[]
				{
					1, omd_fpn_feat_0.Dimensions[1], omd_fpn_feat_0.Dimensions[2] / 4, omd_fpn_feat_0.Dimensions[3] / 4
				});

		var promptlen = _text_encoder.InputMetadata["input_ids"].Dimensions[1];
		var alt_text_features =
			new DenseTensor<float>(text_features.Buffer[0..(promptlen * 256)],
				new[] { 1, promptlen, 256 });

		var alt_text_mask =
			new DenseTensor<bool>(text_mask.Buffer[0..promptlen],
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
		dt_pred_masks = results[0].AsTensor<float>().ToDenseTensor();
		dt_pred_boxes = results[1].AsTensor<float>().ToDenseTensor();
		dt_pred_logits = results[2].AsTensor<float>().ToDenseTensor();
		dt_presence_logits = results[3].AsTensor<float>().ToDenseTensor();
	}

	private DenseTensor<float> dt_pred_masks;
	private DenseTensor<float> dt_pred_boxes; //这个是四个坐标 对应原有的长宽 x1 y1 x2 y2 要乘以1008
	private DenseTensor<float> dt_pred_logits; //这个是个数组，每个乘以下面的是置信度
	private DenseTensor<float> dt_presence_logits; //这个应该只有一个值，是个分数

	public static float Sigmoid(double value)
	{
		return 1.0f / (1.0f + (float)Math.Exp(-value));
	}

	public int LimitRange(int v, int min, int max)
	{
		if (v <= min)
			return min;
		if (v >= max)
			return max - 1;
		return v;
	}

	public void PostProcessor()
	{
		List<(RectangleF, float)> resultBoxs = new List<(RectangleF, float)>();
		var presence_score = Sigmoid(dt_presence_logits.GetValue(0));
		var boxdata = dt_pred_boxes.Buffer;
		for (int i = 0; i < dt_pred_logits.Length; i++)
		{
			float score = Sigmoid(dt_pred_logits.GetValue(i)) * presence_score;
			if (score <= 0.4f)
				continue;
			var x1 = (int)(boxdata.Span[i * 4 + 0] * 1008);
			var y1 = (int)(boxdata.Span[i * 4 + 1] * 1008);
			var x2 = (int)(boxdata.Span[i * 4 + 2] * 1008);
			var y2 = (int)(boxdata.Span[i * 4 + 3] * 1008);

			x1 = LimitRange(x1, 0, 1008);
			y1 = LimitRange(y1, 0, 1008);
			x2 = LimitRange(x2, 0, 1008);
			y2 = LimitRange(y2, 0, 1008);
			if (x2 <= x1)
				continue;
			if (y2 <= y1)
				continue;
			var rc = new RectangleF(x1, y1, x2 - x1, y2 - y1);
			resultBoxs.Add((rc, score));
		}

		resultBoxs.Sort((l, r) => l.Item2.CompareTo(r.Item2));
		clonedimg.Mutate(x =>
		{
			var pen = Pens.Solid(Color.White, 5);
			foreach (var box in resultBoxs)
			{
				x.Draw(pen, box.Item1);
			}
		});
		clonedimg.SaveAsJpeg("d:\\heap.jpg");
		return;
	}


	private Image<Rgb24> clonedimg;

	public void EncodeImage(Image<Rgb24> img)
	{
		original_image_sizes_ = (img.Width, img.Height);
		var destpos = CalcResizeInfo(img.Width, img.Height);
		clonedimg = img.Clone(x =>
		{
			x.Resize(new ResizeOptions
			{
				Mode = ResizeMode.Manual,
				Position = AnchorPositionMode.TopLeft,
				Size = new Size(input_image_width_, input_image_height_),
				Compand = false,
				PadColor = Color.White,
				TargetRectangle = new Rectangle(0, 0, destpos.Item1, destpos.Item2)
			});
		});
		float[] inputTensorValues = new float[input_image_width_ * input_image_height_ * 3];

		var memtensor = new MemoryTensor<float>(inputTensorValues, _vision_encoder.InputMetadata["images"].Dimensions);
		PixelsNormalizer.NormalizerPixelsToTensor(clonedimg, memtensor, new Vector<int>(0, 0));
		var inputTensor =
			new DenseTensor<float>(inputTensorValues, new[] { 1, 3, input_image_width_, input_image_height_ });
		using var results = _vision_encoder.Run(new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("images", inputTensor),
		});
		fpn_feat_0 = results[0].AsTensor<float>().ToDenseTensor();
		fpn_feat_1 = results[1].AsTensor<float>().ToDenseTensor();
		fpn_feat_2 = results[2].AsTensor<float>().ToDenseTensor();
		fpn_pos_2 = results[3].AsTensor<float>().ToDenseTensor();
		return;
	}

	public void EncodeText(string txt)
	{
		var ids = _tokenizer.Encode(txt).ToList();
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
		text_features = results.First().AsTensor<float>().ToDenseTensor();
		text_mask = results[1].AsTensor<bool>().ToDenseTensor();
		return;
	}

	public (int, int) CalcResizeInfo(int width, int height)
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

	public void preprocess(Sam3Input input)
	{
		var img = input.image;
		original_image_sizes_ = (img.Width, img.Height);
		var destpos = CalcResizeInfo(img.Width, img.Height);
		using var clonedimg = img.Clone(x =>
		{
			x.Resize(new ResizeOptions
			{
				Mode = ResizeMode.Manual,
				Position = AnchorPositionMode.TopLeft,
				Size = new Size(1008, 1008),
				Compand = false,
				PadColor = Color.White,
				TargetRectangle = new Rectangle(0, 0, destpos.Item1, destpos.Item2)
			});
		});
		byte[] inputTensorValues = new byte[input_image_width_ * input_image_height_ * 3];
		clonedimg.CopyPixelDataTo(inputTensorValues);
		var inputTensor =
			new DenseTensor<byte>(inputTensorValues, new[] { 1, 3, input_image_width_, input_image_height_ });
	}
}