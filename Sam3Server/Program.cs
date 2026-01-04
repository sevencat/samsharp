using NLog.Extensions.Logging;
using Sam3Sharp;
using SixLabors.ImageSharp;

namespace Sam3Server;

public class Program
{
	public static void Main(string[] args)
	{
		Configuration.Default.PreferContiguousImageBuffers = true;
		var builder = WebApplication.CreateBuilder(args);
		builder.Logging.ClearProviders().AddNLog();

		var config = builder.Configuration;

		builder.Services.AddControllers();
		builder.Services.AddOpenApi();

		var sam3infer = CreateSam3Infer(config);
		builder.Services.AddSingleton(sam3infer);

		var app = builder.Build();

		app.MapOpenApi();
		app.MapControllers();
		app.Run();
	}

	public static Sam3Infer CreateSam3Infer(IConfiguration config)
	{
		var sam3cfg = config.GetSection("sam3").Get<Sam3InferConfig>();
		return new Sam3Infer(sam3cfg);
	}
}