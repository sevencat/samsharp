using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace Sam3Sharp;

public class Sam3Session : IDisposable
{
	public void Dispose()
	{
		org_image?.Dispose();
		input_image?.Dispose();
	}

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

	public RectangleF Transform(RectangleF rc, float width_ratio, float height_ratio, int maxwidth, int maxheight)
	{
		var x = rc.X * width_ratio;
		if (x > maxwidth)
			x = maxwidth;
		var y = rc.Y * height_ratio;
		if (y > maxheight)
			y = maxheight;
		var w = rc.Width * width_ratio;
		if (w > maxwidth)
			w = maxwidth;
		var h = rc.Height * height_ratio;
		if (h > maxheight)
			h = maxheight;
		return new RectangleF(x, y, w, h);
	}

	//不进行复制,直接原图
	public void PlotOrgImage(Sam3Result result)
	{
		//0,0,emb_image_width,org_image_height 这个转换到 0,0,org_image_width,org_image_height
		org_image.Mutate(x =>
		{
			var pen = Pens.Solid(Color.White, 5);
			var width_ratio = (float)org_image.Width / emb_image_width;
			var height_ratio = (float)org_image.Height / emb_image_height;
			foreach (var item in result.items)
			{
				var box = item.box;
				var newbox = Transform(box, width_ratio, height_ratio, org_image_width, org_image_height);
				x.Draw(pen, newbox);
				//画透明层
			}
		});

		//来画掩码!!!
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

		//缩放比例用的长边
		var calclen = Math.Max(org_image_width, org_image_height);
		var calcratio = 288.0f / calclen;

		org_image.DangerousTryGetSinglePixelMemory(out var orgimgdat);
		var orgimgspan = orgimgdat.Span;
		for (int y = 0; y < org_image.Height; y++)
		{
			var curmasky = (int)(y * calcratio);
			for (int x = 0; x < org_image.Width; x++)
			{
				var curmaskx = (int)(x * calcratio);
				var maskpos = curmasky * 288 + curmaskx;
				var maskpixel = imgmaskspan[maskpos].PackedValue;
				//这个是在原画中的位置坐标
				var pos = y * org_image.Width + x;
				if (maskpixel > 100)
				{
					var transcolor = orgimgspan[pos];
					var gray = (299 * transcolor.R + 587 * transcolor.G + 114 * transcolor.B + 500) / 1000;
					transcolor.R = 0;
					transcolor.G = 0;
					transcolor.B = (byte)gray;
					orgimgspan[pos] = transcolor;
				}
			}
		}
	}

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