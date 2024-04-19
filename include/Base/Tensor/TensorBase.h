#pragma once
#include <unordered_map>
#include "Value.h"

#define LibSvcUnSupportedTypeException LibSvcThrow("UnSupported Type!")

//Result = ::libsvc::Float16::OperatorFunction(__VA_ARGS__)
#define LibSvcOperator(Result, OperatorFunction, ...) { \
	if(this->DType_ == ::libsvc::TensorType::Boolean) \
		Result = ::libsvc::Int8::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Int8) \
		Result = ::libsvc::Int8::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Int16) \
		Result = ::libsvc::Int16::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Int32) \
		Result = ::libsvc::Int32::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Int64) \
		Result = ::libsvc::Int64::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Float16) \
		LibSvcUnSupportedTypeException \
	else if(this->DType_ == ::libsvc::TensorType::Float32) \
		Result = ::libsvc::Float32::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Float64) \
		Result = ::libsvc::Float64::OperatorFunction(__VA_ARGS__) \
	else if(this->DType_ == ::libsvc::TensorType::Complex32) \
		Result = ::libsvc::Complex32::OperatorFunction(__VA_ARGS__) \
	else \
		LibSvcUnSupportedTypeException \
}

LibSvcBegin
using SizeType = int64;

enum class TensorType
{
	Boolean,
	Complex32,
	Float16,
	Float32,
	Float64,
	Int8,
	Int16,
	Int32,
	Int64
};

int64 DType2Size(TensorType _Type);
int64 Type2Size(const std::string& _Type);
const std::string& DType2Type(TensorType _Type);
TensorType Type2DType(const std::string& _Type);

class TensorBase : public Value
{
public:
	TensorBase() = delete;
	TensorBase(TensorType _DType);
	~TensorBase() override = default;

protected:
	TensorType DType_ = TensorType::Float32;

public:
	TensorType DType() const;
};

LibSvcEnd