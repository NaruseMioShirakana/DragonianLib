#pragma once

#define DragonianLibUnSupportedTypeException DragonianLibThrow("UnSupported Type!")

//Result = ::DragonianLib::Float16::OperatorFunction(__VA_ARGS__)
#define DragonianLibOperator(DType ,Result, OperatorFunction, ...) { \
	if(DType == ::DragonianLib::TensorType::Boolean) \
		Result = ::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int8) \
		Result = ::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int16) \
		Result = ::DragonianLib::Int16::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int32) \
		Result = ::DragonianLib::Int32::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int64) \
		Result = ::DragonianLib::Int64::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Float16) \
		DragonianLibUnSupportedTypeException; \
	else if(DType == ::DragonianLib::TensorType::Float32) \
		Result = ::DragonianLib::Float32::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Float64) \
		Result = ::DragonianLib::Float64::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Complex32) \
		DragonianLibUnSupportedTypeException; \
	else \
		DragonianLibUnSupportedTypeException; \
}

#define DragonianLibOperatorNoRetrun(OperatorFunction, ...) { \
	if(this->DType_ == ::DragonianLib::TensorType::Boolean) \
		::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Int8) \
		::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Int16) \
		::DragonianLib::Int16::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Int32) \
		::DragonianLib::Int32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Int64) \
		::DragonianLib::Int64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Float16) \
		DragonianLibUnSupportedTypeException; \
	else if(this->DType_ == ::DragonianLib::TensorType::Float32) \
		::DragonianLib::Float32::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Float64) \
		::DragonianLib::Float64::OperatorFunction(__VA_ARGS__); \
	else if(this->DType_ == ::DragonianLib::TensorType::Complex32) \
		DragonianLibUnSupportedTypeException; \
	else \
		DragonianLibUnSupportedTypeException; \
}

#define DragonianLibOperatorDTypeNoRetrun(DType, OperatorFunction, ...) { \
	if(DType == ::DragonianLib::TensorType::Boolean) \
		::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int8) \
		::DragonianLib::Int8::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int16) \
		::DragonianLib::Int16::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int32) \
		::DragonianLib::Int32::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Int64) \
		::DragonianLib::Int64::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Float16) \
		DragonianLibUnSupportedTypeException; \
	else if(DType == ::DragonianLib::TensorType::Float32) \
		::DragonianLib::Float32::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Float64) \
		::DragonianLib::Float64::OperatorFunction(__VA_ARGS__); \
	else if(DType == ::DragonianLib::TensorType::Complex32) \
		DragonianLibUnSupportedTypeException; \
	else \
		DragonianLibUnSupportedTypeException; \
}

#define DragonianLibTypeSwitch(_CDType, _Boolean, _Int8, _Int16, _Int32, _Int64, _Float16, _Float32, _Float64, _Complex32) { \
	if((_CDType) == ::DragonianLib::TensorType::Boolean) \
		{_Boolean} \
	else if((_CDType) == ::DragonianLib::TensorType::Int8) \
		{_Int8} \
	else if((_CDType) == ::DragonianLib::TensorType::Int16) \
		{_Int16} \
	else if((_CDType) == ::DragonianLib::TensorType::Int32) \
		{_Int32} \
	else if((_CDType) == ::DragonianLib::TensorType::Int64) \
		{_Int64} \
	else if((_CDType) == ::DragonianLib::TensorType::Float16) \
		DragonianLibUnSupportedTypeException; \
	else if((_CDType) == ::DragonianLib::TensorType::Float32) \
		{_Float32} \
	else if((_CDType) == ::DragonianLib::TensorType::Float64) \
		{_Float64} \
	else if((_CDType) == ::DragonianLib::TensorType::Complex32) \
		DragonianLibUnSupportedTypeException; \
	else \
		DragonianLibUnSupportedTypeException; \
}

#define DragonianLibCastImpl(_DST_TYPE, _DST_VAL, _SRC_TYPE, _SRC_VAL) { \
	_SRC_TYPE DragonianLib_CAST_TEMP = *(const _SRC_TYPE*)(_SRC_VAL); \
	(_DST_VAL) = (_DST_TYPE)DragonianLib_CAST_TEMP; \
}

#define DragonianLibCastComplexImpl(_DST_TYPE, _DST_VAL, _SRC_VAL) { \
	const auto & _Cpx = *(std::complex<float>*)(_SRC_VAL); \
	(_DST_VAL) = (_DST_TYPE)sqrt(_Cpx.real() * _Cpx.real() + _Cpx.imag() * _Cpx.imag()); \
}

#define DragonianLibCycle(IndicesPtr, ShapePtr, CurDims, Body) while ((IndicesPtr)[0] < (ShapePtr)[0])\
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