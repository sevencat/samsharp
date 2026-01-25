using Microsoft.AspNetCore.Mvc;
using Sam3Server.Entity;
using Sam3Sharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sam3Server.Controllers;

[ApiController]
[Route("/api/infer")]
public class InferController(Sam3Infer sam3Infer)
{
	[HttpPost("sam3imgret")]
	public IActionResult Sam3ImageRet(IFormFile formFile, [FromQuery] string prompt)
	{
		using var img = Image.Load(formFile.OpenReadStream());
		using var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = prompt;
		var result = sam3Infer.Handle(session);
		session.PlotOrgImage(result);
		using var ms = new MemoryStream();
		session.org_image.SaveAsJpeg(ms);
		return new FileContentResult(ms.ToArray(), "image/jpeg");
	}

	[HttpPost("sam3ret")]
	public CommonResult<Sam3Result> Sam3Result(IFormFile formFile, [FromQuery] string prompt)
	{
		using var img = Image.Load(formFile.OpenReadStream());
		using var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = prompt;
		var result = sam3Infer.Handle(session);
		return CommonResult.Success(result);
	}

	[HttpPost("sam3detectret")]
	public CommonResult<Sam3Result> Sam3Detect(IFormFile formFile, [FromQuery] string prompt)
	{
		using var img = Image.Load(formFile.OpenReadStream());
		using var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = prompt;
		var result = sam3Infer.Handle(session);
		foreach (var item in result.items)
			item.mask = null;
		return CommonResult.Success(result);
	}
}