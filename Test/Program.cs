using OpenCvSharp;
using Sam3Sharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Size = OpenCvSharp.Size;

namespace Test;

class Program
{
	static void Main(string[] args)
	{
		Configuration.Default.PreferContiguousImageBuffers = true;
		var modeldir = @"F:\zb\flycheck\rsam3\model";
		var cfg = new Sam3InferConfig();
		cfg.vision_encoder_path = Path.Combine(modeldir, "vision-encoder-fp16.onnx");
		cfg.text_encoder_path = Path.Combine(modeldir, "text-encoder-fp16.onnx");
		cfg.decoder_path = Path.Combine(modeldir, "decoder-fp16.onnx");
		cfg.tokenizer_path = Path.Combine(modeldir, "tokenizer.json");
		
		var sam3 = new Sam3Infer(cfg);
		
		var imagePath = Path.Combine(modeldir, "persons.jpg");

		var img = Image.Load(imagePath);
		var img2 = img.CloneAs<Rgb24>();
		sam3.EncodeImage(img2);
		sam3.EncodeText("person");
		sam3.Decode();

		sam3.Gc();
		sam3.PostProcessor();
		return;
	}

	public static (int, int) calc(int width, int height)
	{
		if (width >= height)
		{
			var ratio = width / 1008m;
			var h = (int)(height / ratio);
			if (h > 1008)
				h = 1008;
			return (1108, h);
		}
		else
		{
			var ratio = height / 1008m;
			var w = (int)(width / ratio);
			if (w > 1008)
				w = 1008;
			return (w, 1008);
		}
	}

	static void Main3(string[] args)
	{
		var modeldir = @"F:\zb\flycheck\rsam3\model";
		var encoderPath = Path.Combine(modeldir, "vision-encoder-fp16.onnx");
		var decoderPath = Path.Combine(modeldir, "decoder-fp16.onnx");
		var imagePath = Path.Combine(modeldir, "x3.jpg");

		var img = Image.Load(imagePath);
		var img2 = img.CloneAs<Rgb24>();
		var destpos = calc(img2.Width, img2.Height);
		img2.Mutate(x =>
		{
			x.Resize(new ResizeOptions
			{
				Mode = ResizeMode.Manual,
				Position = AnchorPositionMode.TopLeft,
				Size = new SixLabors.ImageSharp.Size(1008, 1008),
				Compand = false,
				PadColor = Color.White,
				TargetRectangle = new Rectangle(0, 0, destpos.Item1, destpos.Item2)
			});
		});
		img2.SaveAsJpeg("d:\\t2.jpg");
		return;

		using var sam = new SamInferenceSession(encoderPath, decoderPath);
		sam.Initialize();


		string windowName = "Segment Anything (Click different parts of image to segment)";
		Console.WriteLine("Setting image...");
		sam.SetImage(imagePath);


		Mat displayImage = new(imagePath);
		Mat baseImage = new(imagePath);
		Window window = new(windowName);
		Cv2.ImShow(windowName, displayImage);

		Cv2.SetMouseCallback(windowName, (mouseEvent, xCoord, yCoord, flags, ptr) =>
		{
			if (mouseEvent == MouseEventTypes.LButtonDown)
			{
				float[] mask = sam.GetPointMask(xCoord, yCoord);

				displayImage.Dispose();
				displayImage = new Mat(new Size(baseImage.Width, baseImage.Height), baseImage.Type());

				int pixel = 0;
				for (int y = 0; y < baseImage.Rows; y++)
				{
					for (int x = 0; x < baseImage.Cols; x++)
					{
						double vibrance = mask[pixel++] > 0 ? 1.0 : 0.5;
						displayImage.At<Vec3b>(y, x) = baseImage.At<Vec3b>(y, x) * vibrance;
					}
				}

				Cv2.ImShow(windowName, displayImage);
			}
		});

		int tmp = Cv2.WaitKey();
	}
}