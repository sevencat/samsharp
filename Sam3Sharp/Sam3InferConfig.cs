namespace Sam3Sharp;

public class Sam3InferConfig
{
	public string vision_encoder_path { get; set; }
	public string text_encoder_path { get; set; }
	public string geometry_encoder_path { get; set; }
	public string decoder_path { get; set; }
	public string tokenizer_path { get; set; }
	public bool use_cuda { get; set; }
}