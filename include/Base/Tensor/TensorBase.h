#pragma once
#include <complex>
#include <unordered_map>
#include "Value.h"

#define LibSvcUnSupportedTypeException LibSvcThrow("UnSupported Type!")

//Result = ::libsvc::Float16::OperatorFunction(__VA_ARGS__)
#define LibSvcOperator(Result, OperatorFunction, ...) { \
	if(this->DType_ == ::libsvc::TensorType::Boolean) \
		Result = ::libsvc::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int8) \
		Result = ::libsvc::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int16) \
		Result = ::libsvc::Int16::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int32) \
		Result = ::libsvc::Int32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int64) \
		Result = ::libsvc::Int64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Float16) \
		LibSvcUnSupportedTypeException \
	else if(this->DType_ == ::libsvc::TensorType::Float32) \
		Result = ::libsvc::Float32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Float64) \
		Result = ::libsvc::Float64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Complex32) \
		LibSvcUnSupportedTypeException \
	else \
		LibSvcUnSupportedTypeException \
}

#define LibSvcOperatorNoRetrun(OperatorFunction, ...) { \
	if(this->DType_ == ::libsvc::TensorType::Boolean) \
		::libsvc::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int8) \
		::libsvc::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int16) \
		::libsvc::Int16::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int32) \
		::libsvc::Int32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Int64) \
		::libsvc::Int64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Float16) \
		LibSvcUnSupportedTypeException \
	else if(this->DType_ == ::libsvc::TensorType::Float32) \
		::libsvc::Float32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Float64) \
		::libsvc::Float64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::libsvc::TensorType::Complex32) \
		LibSvcUnSupportedTypeException \
	else \
		LibSvcUnSupportedTypeException \
}

#define LibSvcTypeSwitch(_CDType, _Boolean, _Int8, _Int16, _Int32, _Int64, _Float16, _Float32, _Float64, _Complex32) { \
	if((_CDType) == ::libsvc::TensorType::Boolean) \
		{_Boolean} \
	else if((_CDType) == ::libsvc::TensorType::Int8) \
		{_Int8} \
	else if((_CDType) == ::libsvc::TensorType::Int16) \
		{_Int16} \
	else if((_CDType) == ::libsvc::TensorType::Int32) \
		{_Int32} \
	else if((_CDType) == ::libsvc::TensorType::Int64) \
		{_Int64} \
	else if((_CDType) == ::libsvc::TensorType::Float16) \
		LibSvcUnSupportedTypeException \
	else if((_CDType) == ::libsvc::TensorType::Float32) \
		{_Float32} \
	else if((_CDType) == ::libsvc::TensorType::Float64) \
		{_Float64} \
	else if((_CDType) == ::libsvc::TensorType::Complex32) \
		LibSvcUnSupportedTypeException \
	else \
		LibSvcUnSupportedTypeException \
}

#define LibSvcCastImpl(_DST_TYPE, _DST_VAL, _SRC_TYPE, _SRC_VAL) { \
	_SRC_TYPE LIBSVC_CAST_TEMP = *(_SRC_TYPE*)(_SRC_VAL); \
	(_DST_VAL) = (_DST_TYPE)LIBSVC_CAST_TEMP; \
}

#define LibSvcCastComplexImpl(_DST_TYPE, _DST_VAL, _SRC_VAL) { \
	const auto & _Cpx = *(std::complex<float>*)(_SRC_VAL); \
	(_DST_VAL) = (_DST_TYPE)sqrt(_Cpx.real() * _Cpx.real() + _Cpx.imag() * _Cpx.imag()); \
}

#define LibSvcCycle(IndicesPtr, ShapePtr, CurDims, Body) while ((IndicesPtr)[0] < (ShapePtr)[0])\
{ \
	{Body} \
	for (SizeType i = (CurDims) - 1; i >= 0; --i) \
	{ \
		++(IndicesPtr)[i]; \
		if ((IndicesPtr)[i] < (ShapePtr)[i]) \
			break; \
		if (i) (IndicesPtr)[i] = 0; \
	} \
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