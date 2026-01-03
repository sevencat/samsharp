using SixLabors.ImageSharp;

namespace Sam3Sharp;

public class Sam3ResultItem
{
	public float score { get; set; }
	public RectangleF box { get; set; }
	public float[] mask { get; set; }
}