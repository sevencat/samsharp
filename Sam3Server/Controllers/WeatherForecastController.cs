using Microsoft.AspNetCore.Mvc;

namespace Sam3Server.Controllers;

[ApiController]
[Route("/api/test")]
public class TestController : ControllerBase
{
	private static readonly string[] Summaries =
	[
		"Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
	];

	[HttpGet(Name = "echo")]
	public string Echo([FromQuery] string message)
	{
		return message;
	}
}