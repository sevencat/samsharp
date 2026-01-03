namespace Sam3Sharp;

public class Sam3Result
{
	public int mask_model_width { get; set; }
	public int mask_model_height { get; set; }

	public List<Sam3ResultItem> items { get; set; } = [];
}