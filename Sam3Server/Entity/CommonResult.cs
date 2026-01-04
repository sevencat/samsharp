namespace Sam3Server.Entity;

public class CommonResult
{
	public int code { get; set; }
	public string msg { get; set; }

	public static CommonResult Create(int code, string msg)
	{
		return new CommonResult()
		{
			code = code,
			msg = msg
		};
	}

	public static CommonResult<T> Create<T>(int code, string msg, T data)
	{
		return new CommonResult<T>()
		{
			code = code,
			msg = msg,
			data = data
		};
	}

	public static CommonResult Success()
	{
		return Create(HttpStatus.SUCCESS, "success");
	}

	public static CommonResult<T> Success<T>(T data, string msg = "success")
	{
		return Create(HttpStatus.SUCCESS, msg, data);
	}

	public static CommonResult Error(ResultCode code, string msg = "")
	{
		return Create((int)code, msg);
	}

	public static CommonResult Error(int code, string msg)
	{
		return Create(code, msg);
	}

	public static CommonResult Error(string msg)
	{
		return Create((int)ResultCode.CUSTOM_ERROR, msg);
	}

	public static CommonResult<T> Error<T>(ResultCode code, string msg = "")
	{
		return Create<T>((int)code, msg, default(T));
	}

	public static CommonResult<T> Error<T>(int code, string msg)
	{
		return Create<T>(code, msg, default(T));
	}

	public static CommonResult<T> Error<T>(string msg)
	{
		return Create<T>((int)ResultCode.CUSTOM_ERROR, msg, default(T));
	}

	public static CommonResult CreateDbUpdate(int affectrows)
	{
		if (affectrows > 0)
			return Success();
		return Error("数据库操作失败");
	}
}

public class CommonResult<T> : CommonResult
{
	public T data { get; set; }
}

public class CommonResultList<T> : CommonResult
{
	public List<T> list { get; set; }
}

public static class CommonResultExt
{
	public static CommonResult<T> ToCommonResult<T>(this T data)
	{
		if (data == null)
			return new CommonResult<T>()
			{
				code = (int)ResultCode.NO_DATA,
			};
		else
			return CommonResult.Success<T>(data);
	}
}