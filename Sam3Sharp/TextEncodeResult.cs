using Microsoft.ML.OnnxRuntime.Tensors;

namespace Sam3Sharp;

public class TextEncodeResult
{
	public int[] text_dimensions { get; set; }
	public string prompt { get; set; }

	public DenseTensor<float> text_features { get; set; }
	public DenseTensor<bool> text_mask { get; set; }
}