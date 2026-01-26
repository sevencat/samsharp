using Microsoft.ML.OnnxRuntime;

namespace Sam3Sharp;

public static class OnnxExt
{
	public static int GetShapeLen(this NodeMetadata metadata)
	{
		var result = 1;
		foreach (var dim in metadata.Dimensions)
			if (dim > 0)
				result = result * dim;
		return result;
	}

	public static int[] CopyDimensions(this NodeMetadata metadata)
	{
		int[] destdim = new int[metadata.Dimensions.Length];
		for (int i = 0; i < metadata.Dimensions.Length; i++)
		{
			int srcdim = metadata.Dimensions[i];
			if (srcdim < 0)
				srcdim = 1;
			destdim[i] = srcdim;
		}

		return destdim;
	}
}