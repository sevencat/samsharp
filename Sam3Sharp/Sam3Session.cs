using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace Sam3Sharp;

public class Sam3Session
{
	public void GCInput()
	{
		text_features = null;
		text_mask = null;
		fpn_feat_0 = null;
		fpn_feat_1 = null;
		fpn_feat_2 = null;
		fpn_pos_2 = null;
		GC.Collect();
	}
	
	public void GCOutput()
	{
		dt_pred_masks = null;
		dt_pred_boxes = null;
		dt_pred_logits = null;
		dt_presence_logits = null;
		GC.Collect();
	}

	public string prompt_text { get; set; }

	//原始图像
	public Image<Rgb24> org_image { get; set; }

	public int org_image_width { get; set; } //原始图片宽度
	public int org_image_height { get; set; } //原始图片高度

	//输入图片,原图片大小resize到模型需要大小(1008*1008)
	public Image<Rgb24> input_image { get; set; }
	public int emb_image_width { get; set; }
	public int emb_image_height { get; set; }

	public DenseTensor<float> text_features { get; set; }
	public DenseTensor<bool> text_mask { get; set; }

	public DenseTensor<float> fpn_feat_0 { get; set; }
	public DenseTensor<float> fpn_feat_1 { get; set; }
	public DenseTensor<float> fpn_feat_2 { get; set; }
	public DenseTensor<float> fpn_pos_2 { get; set; }


	public DenseTensor<float> dt_pred_masks { get; set; } //200*288*288 200是最多200个结果,288*288是把原图弄成288 288 ,需要放大
	public DenseTensor<float> dt_pred_boxes { get; set; } //这个是四个坐标 对应原有的长宽 x1 y1 x2 y2 要乘以1008
	public DenseTensor<float> dt_pred_logits { get; set; } //这个是个数组，每个乘以下面的是置信度
	public DenseTensor<float> dt_presence_logits { get; set; } //这个应该只有一个值，是个分数

	public void PlotInputImage(Sam3Result result)
	{
		//开始画框
		input_image.Mutate((x =>
		{
			var pen = Pens.Solid(Color.White, 5);
			foreach (var item in result.items)
			{
				var box = item.box;
				x.Draw(pen, box);
				//画透明层
			}
		}));
		//开始进行画掩码

		//生成一个掩码图用来合并
		using var imgmask = new Image<L8>(288, 288, new L8(0));
		imgmask.DangerousTryGetSinglePixelMemory(out var imgmaskdat);
		var imgmaskspan = imgmaskdat.Span;
		foreach (var item in result.items)
		{
			var curmask = new Span<float>(item.mask);
			for (var idx = 0; idx < 288 * 288; idx++)
			{
				var curmaskvalue = curmask[idx];

				if (curmaskvalue >= 0.5)
				{
					imgmaskspan[idx].PackedValue = 255;
				}
			}
		}

		imgmask.Mutate(x => x.Resize(input_image.Width, input_image.Height, new NearestNeighborResampler()));
		imgmask.DangerousTryGetSinglePixelMemory(out var imgmask2ptr);
		var imgmask2span = imgmask2ptr.Span;
		
		input_image.DangerousTryGetSinglePixelMemory(out var resultmemptr);
		var resultimgspan = resultmemptr.Span;
		for (int y = 0; y < input_image.Height; y++)
		{
			for (int x = 0; x < input_image.Width; x++)
			{
				var pos = y * input_image.Width + x;
				var maskpixel = imgmask2span[pos].PackedValue;
				if (maskpixel > 100)
					resultimgspan[pos] = new Rgb24(255, 255, 0);
			}
		}
	}
}