using NLog.Extensions.Logging;
using SixLabors.ImageSharp;

namespace Sam3Server;

public class Program
{
	public static void Main(string[] args)
	{
		Configuration.Default.PreferContiguousImageBuffers = true;
		var builder = WebApplication.CreateBuilder(args);
		builder.Logging.ClearProviders().AddNLog();

		builder.Services.AddControllers();
		builder.Services.AddOpenApi();

		var app = builder.Build();

		app.MapOpenApi();
		app.MapControllers();
		app.Run();
	}
}