using OpenCvSharp;

namespace Sam3Sharp;

public enum ObjectType
{
	UNKNOW = -1,
	POSITION = 0,
	POSE = 1,
	OBB = 2,
	SEGMENTATION = 3,
	DEPTH_ANYTHING = 4,
	DEPTH_PRO = 5,
	TRACK = 6,
	DETECTION = 7,
};

public class Box
{
	public float left { get; set; } = 0.0f;
	public float top { get; set; } = 0.0f;
	public float right { get; set; } = 0.0f;
	public float bottom { get; set; } = 0.0f;
}

public class Pose
{
	public List<PosePoint> points { get; set; }
}

public class PosePoint
{
	public float x { get; set; } = 0.0f;
	public float y { get; set; } = 0.0f;
	public float vis { get; set; } = 0.0f;
}

public class Obb
{
	public float cx { get; set; } = 0.0f;
	public float cy { get; set; } = 0.0f;
	public float w { get; set; } = 0.0f;
	public float h { get; set; } = 0.0f;
	public float angle { get; set; } = 0.0f;
}

public class SegmentMap
{
	public int width { get; set; } = 0;
	public int height { get; set; } = 0;
	public List<byte> data { get; set; }
}

public class Segmentation
{
	public Mat mask { get; set; }
}

public class DetectionBox
{
	public ObjectType type { get; set; } = ObjectType.UNKNOW;
	public Box box { get; set; }
	public float score { get; set; } = 0.0f;
	public int class_id { get; set; } = -1;
	public string class_name { get; set; }

	// --- 可选的数据 ---
	public Pose pose { get; set; }
	public Obb obb { get; set; }
	public Segmentation segmentation { get; set; }
}