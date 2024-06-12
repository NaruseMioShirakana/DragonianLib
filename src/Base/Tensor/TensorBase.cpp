#include "Tensor/TensorBase.h"

DragonianLibSpaceBegin
const std::unordered_map<std::string, TensorType> __Ty2DTy{
	{ "int8", TensorType::Int8},
	{ "int16", TensorType::Int16 },
	{ "int32", TensorType::Int32 },
	{ "int64", TensorType::Int64 },
	{ "float16", TensorType::Float16 },
	{ "float32", TensorType::Float32 },
	{ "float64", TensorType::Float64 },
	{ "complex32", TensorType::Complex32 },
	{ "bool", TensorType::Boolean }
};

const std::unordered_map<TensorType, std::string> __DTy2Ty{
	{ TensorType::Int8, "int8" },
	{ TensorType::Int16, "int16" },
	{ TensorType::Int32, "int32" },
	{ TensorType::Int64, "int64" },
	{ TensorType::Float16, "float16" },
	{ TensorType::Float32, "float32" },
	{ TensorType::Float64, "float64" },
	{ TensorType::Complex32, "complex32" },
	{ TensorType::Boolean, "bool" }
};

const std::unordered_map<std::string, int64> __Ty2Size{
	{ "int8", sizeof(int8) },
	{ "int16", sizeof(int16) },
	{ "int32", sizeof(int32) },
	{ "int64", sizeof(int64) },
	{ "float16", 2 },
	{ "float32", sizeof(float32) },
	{ "float64", sizeof(float64) },
	{ "complex32", sizeof(std::complex<float32>) },
	{ "bool", sizeof(bool) }
};

const std::unordered_map<TensorType, int64> __DTy2Size{
	{ TensorType::Int8, sizeof(int8) },
	{ TensorType::Int16, sizeof(int16) },
	{ TensorType::Int32, sizeof(int32) },
	{ TensorType::Int64, sizeof(int64) },
	{ TensorType::Float16, 2 },
	{ TensorType::Float32, sizeof(float32) },
	{ TensorType::Float64, sizeof(float64) },
	{ TensorType::Complex32, sizeof(std::complex<float32>) },
	{ TensorType::Boolean, sizeof(bool) }
};

TensorType Type2DType(const std::string& _Type)
{
	const auto res = __Ty2DTy.find(_Type);
	if (res != __Ty2DTy.end())
		return res->second;
	DragonianLibThrow("Type Error!");
}

const std::string& DType2Type(TensorType _Type)
{
	const auto res = __DTy2Ty.find(_Type);
	if (res != __DTy2Ty.end())
		return res->second;
	DragonianLibThrow("Type Error!");
}

int64 Type2Size(const std::string& _Type)
{
	const auto res = __Ty2Size.find(_Type);
	if (res != __Ty2Size.end())
		return res->second;
	DragonianLibThrow("Type Error!");
}

int64 DType2Size(TensorType _Type)
{
	const auto res = __DTy2Size.find(_Type);
	if (res != __DTy2Size.end())
		return res->second;
	DragonianLibThrow("Type Error!");
}

TensorType TensorBase::DType() const
{
	return DType_;
}

TensorBase::TensorBase(TensorType _DType)
{
	DType_ = _DType;
}

DragonianLibSpaceEnd