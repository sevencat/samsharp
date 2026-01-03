namespace Sam3Sharp;

public class Sam3Result
{
	public int model_width { get; set; }
	public int model_height { get; set; }

	public List<Sam3ResultItem> items { get; set; } = [];
}