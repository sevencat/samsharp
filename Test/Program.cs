using Sam3Sharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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

		var imagePath = Path.Combine(modeldir, "x4.jpg");

		var img = Image.Load(imagePath);
		var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = "heap";

		var result = sam3.Handle(session);
		session.PlotOrgImage(result);
		session.org_image.SaveAsJpeg("d:\\ret.jpg");
		return;
	}
}