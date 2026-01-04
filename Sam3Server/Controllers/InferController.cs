using Microsoft.AspNetCore.Mvc;

namespace Sam3Server.Controllers;

[ApiController]
[Route("/api/infer")]
public class InferController
{
	public IActionResult Sam3(IFormFile formFile, [FromQuery] string prompt)
	{
		
	}
}