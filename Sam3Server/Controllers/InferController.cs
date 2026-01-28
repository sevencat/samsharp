using Microsoft.AspNetCore.Mvc;
using Sam3Server.Entity;
using Sam3Sharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Sam3Server.Controllers;

public class Sam3Result2
{
	public Sam3Result result { get; set; }
	public string base64img { get; set; }
}

[ApiController]
[Route("/api/infer")]
public class InferController(Sam3Infer sam3Infer)
{
	[HttpPost("sam3imgret2")]
	public CommonResult<Sam3Result2> Sam3ImageRet2(IFormFile formFile, [FromQuery] string prompt)
	{
		using var img = Image.Load(formFile.OpenReadStream());
		using var session = new Sam3Session();
		session.org_image = img.CloneAs<Rgb24>();
		session.org_image_height = img.Height;
		session.org_image_width = img.Width;
		session.prompt_text = prompt;
		var result = sam3Infer.Handle(session);
		session.PlotOrgImage(result);
		foreach (var item in result.items)
			item.mask = null;
		session.org_image.Mutate(x => { x.Resize(640, 0); });
		using var ms = new MemoryStream();
		session.org_image.SaveAsJpeg(ms);
		Sam3Result2 realresult = new Sam3Result2();
		realresult.result = result;
		realresult.base64img = Convert.ToBase64String(ms.ToArray());
		return CommonResult.Success(realresult);
	}

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