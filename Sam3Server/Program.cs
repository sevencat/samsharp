using System.Text;
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
		app.MapGet("openapiui", (() => Results.Text(RedocHtml, "text/html", Encoding.UTF8)));
		app.MapControllers();
		app.Run();
	}

	public static Sam3Infer CreateSam3Infer(IConfiguration config)
	{
		var sam3cfg = config.GetSection("sam3").Get<Sam3InferConfig>();
		return new Sam3Infer(sam3cfg);
	}

	private const string RedocHtml = """

	                                 <!doctype html> <!-- Important: must specify -->
	                                 <html>
	                                   <head>
	                                     <meta charset="utf-8"> <!-- Important: rapi-doc uses utf8 characters -->
	                                     <script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
	                                   </head>
	                                   <body>
	                                     <rapi-doc server-url="http://localhost:9040/" spec-url = "/openapi/v1.json"> </rapi-doc>
	                                   </body>
	                                 </html>
	                                 """;
}