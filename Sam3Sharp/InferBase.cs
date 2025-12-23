namespace Sam3Sharp;

public class InferResult : List<DetectionBox>
{
}

public class InferResultArray : List<InferResult>
{
}

public abstract class InferBase
{
	// 批量推理 (Core API)
	public abstract InferResultArray forwards(List<Sam3Input> inputs, bool return_mask = false,
		List<byte> stream = null);

	// 单个推理 (Wrapper)
	public InferResult forward(Sam3Input input, bool return_mask = false, List<byte> stream = null)
	{
		return forwards([input], return_mask, stream)[0];
	}

	// 预设文本 Token (用于 Tokenizer 缓存),input_ids 32个长度 attention_mask也是32个长度
	public abstract void setup_text_inputs(string input_text, List<long> input_ids, List<long> attention_mask);
}