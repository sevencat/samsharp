using Microsoft.AspNetCore.Mvc;

namespace Sam3Server.Controllers;

[ApiController]
[Route("/api/test")]
public class TestController : ControllerBase
{
	[HttpGet(Name = "echo")]
	public string Echo([FromQuery] string message)
	{
		return message;
	}
}