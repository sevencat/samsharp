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
		cfg.model_path = modeldir;
		cfg.use_cuda = false;
		var sam3 = new Sam3Infer(cfg);
		

		var imagePath = Path.Combine(modeldir, "x7.png");

		var img = Image.Load(imagePath);
		var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = "person";

		var result = sam3.Handle(session);
		session.PlotOrgImage(result);
		session.org_image.SaveAsJpeg("d:\\ret.jpg");
		return;
	}
}