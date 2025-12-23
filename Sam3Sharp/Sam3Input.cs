using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sam3Sharp;

public class BoxPrompt
{
	public string text { get; set; }
	public float x1 { get; set; }
	public float y1 { get; set; }
	public float x2 { get; set; }
	public float y2 { get; set; }
}

public class Sam3PromptUnit
{
	public string text { get; set; }
	public List<BoxPrompt> boxes { get; set; }
}

public class Sam3Input
{
	public float confidence_threshold { get; set; }
	public Image<Rgb24> image { get; set; }
	public List<Sam3PromptUnit> prompts { get; set; }
}