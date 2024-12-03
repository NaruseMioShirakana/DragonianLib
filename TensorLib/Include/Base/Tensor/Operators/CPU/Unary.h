﻿#pragma once
#include "CPU.h"
#include "Libraries/Util/Logger.h"

#define _D_Dragonian_Lib_Operator_Unary_Function_Def(_Function, Unfold) namespace UnaryOperators { namespace _Function##Unary { \
 \
template <typename Type> \
constexpr bool HasOperatorValue = decltype(IsInvokableWith::CheckConst(_Function##<Type>, InstanceOf<Type>()))::value; \
 \
constexpr auto _D_Dragonian_Lib_Operator_Unary_Unfold = Unfold; \
template<typename _Type>  \
void UnaryInplaceCont##_Function( \
	_Type* _Dest, \
	SizeType DestSize, \
	void* \
) \
{ \
	constexpr int64_t OpThroughput = 2; \
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type)); \
	constexpr int64_t LoopStride = OpThroughput * Stride; \
 \
	SizeType i = 0; \
 \
	bool EnableAvx = IsAvx256SupportedValue<_Type>; \
	try \
	{ \
		if constexpr (IsAvx256SupportedValue<_Type>) \
			_Function(Vectorized<_Type>(_Dest + i)); \
	} \
	catch (std::exception&) \
	{ \
		LogWarn(L"Avx Is Not Supported, Falling Back To Scalar Mode"); \
		EnableAvx = false; \
	} \
 \
	if constexpr (IsAvx256SupportedValue<_Type>) \
	{ \
		if (EnableAvx) \
			for (; i < DestSize - LoopStride; i += LoopStride) \
			{ \
				auto _Dest1 = Vectorized<_Type>(_Dest + i); \
				auto _Dest2 = Vectorized<_Type>(_Dest + i + Stride); \
				auto _Result1 = _Function(_Dest1); \
				auto _Result2 = _Function(_Dest2); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Dest[i + j] = _Function(_Dest[i + j]); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Dest[i + j] = _Function(_Dest[i + j]); \
 \
	for (; i < DestSize; ++i) \
		_Dest[i] = _Function(_Dest[i]); \
} \
 \
template<typename _Type, size_t _NRank> \
void UnaryInplace##_Function( \
	_Type* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	void* \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
 \
	const auto Func = [&](int64_t _Index) \
		{ \
			_Dest[_Index] = _Function(_Dest[_Index]); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
 \
	SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Unary_Unfold>( \
		0, Shape, Begin, ViewStride, Func \
	); \
} \
 \
template<typename _Type> \
void UnaryInplaceContEmpty##_Function( \
	_Type* _Dest, \
	SizeType DestSize, \
	void* \
) {} \
 \
template<typename _Type> \
void UnaryCont##_Function( \
	_Type* _Dest, \
	const _Type* _Src, \
	SizeType DestSize, \
	void* \
) \
{ \
	constexpr int64_t OpThroughput = 2; \
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type)); \
	constexpr int64_t LoopStride = OpThroughput * Stride; \
 \
	SizeType i = 0; \
 \
	bool EnableAvx = IsAvx256SupportedValue<_Type>; \
	try \
	{ \
		if constexpr (IsAvx256SupportedValue<_Type>) \
			_Function(Vectorized<_Type>(_Dest + i)); \
	} \
	catch (std::exception&) \
	{ \
		LogWarn(L"Avx Is Not Supported, Falling Back To Scalar Mode"); \
		EnableAvx = false; \
	} \
 \
	if constexpr (IsAvx256SupportedValue<_Type>) \
	{ \
		if (EnableAvx) \
			for (; i < DestSize - LoopStride; i += LoopStride) \
			{ \
				auto _Src1 = Vectorized<_Type>(_Src + i); \
				auto _Src2 = Vectorized<_Type>(_Src + i + Stride); \
				auto _Result1 = _Function(_Src1); \
				auto _Result2 = _Function(_Src2); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Dest[i + j] = _Function(_Src[i + j]); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Dest[i + j] = _Function(_Src[i + j]); \
 \
	for (; i < DestSize; ++i) \
		_Dest[i] = _Function(_Src[i]); \
} \
 \
template<typename _DestType, typename _SrcType> \
void UnaryContEmpty##_Function ( \
	_DestType* _Dest, \
	const _SrcType* _Src, \
	SizeType DestSize, \
	void* \
) \
{ \
	constexpr int64_t OpThroughput = 8; \
	SizeType i = 0; \
	for (; i < DestSize - OpThroughput; i += OpThroughput) \
		for (int64_t j = 0; j < OpThroughput; ++j) \
			_Dest[i + j] = _Function(_Src[i + j]); \
			 \
	for (; i < DestSize; ++i) \
		_Dest[i] = _Function(_Src[i]); \
} \
 \
template<typename _DestType, typename _SrcType, size_t _NRank> \
void Unary##_Function( \
	_DestType* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	const _SrcType* _Src, \
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld, \
	void* \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
	const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld; \
 \
	const auto Func = [&](int64_t _IndexA, int64_t _IndexB) \
		{ \
			_Dest[_IndexA] = _Function(_Src[_IndexB]); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
	const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data(); \
 \
	DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Unary_Unfold>( \
		0, 0, \
		Shape, Begin, \
		ViewStride, SrcViewStride, \
		Func \
	); \
} \
} \
} \
template <typename _Type> \
template<typename _ResultType, size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Unary( \
	_ResultType* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	bool Continuous \
) \
{ \
	try \
	{ \
		_ResultType Test = UnaryOperators::_Function(*_Src); \
	} \
	catch (std::exception& e) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(e.what()); \
	} \
	if constexpr (IsSameTypeValue<_Type, _ResultType>) \
	{ \
		if (_Dest == _Src) \
			ImplMultiThreadSingle( \
				_Dest, \
				_DestInfo, \
				nullptr, \
				Continuous, \
				UnaryOperators::_Function##Unary::UnaryInplace##_Function<_Type, _NRank>, \
				UnaryOperators::_Function##Unary::UnaryInplaceCont##_Function<_Type> \
			); \
		else \
			ImplMultiThreadDouble( \
				_Dest, \
				_DestInfo, \
				_Src, \
				_SrcInfo, \
				nullptr, \
				Continuous, \
				UnaryOperators::_Function##Unary::Unary##_Function<_ResultType, _Type, _NRank>, \
				UnaryOperators::_Function##Unary::UnaryCont##_Function<_Type> \
			); \
	} \
	else \
	{ \
		ImplMultiThreadDouble( \
			_Dest, \
			_DestInfo, \
			_Src, \
			_SrcInfo, \
			nullptr, \
			Continuous, \
			UnaryOperators::_Function##Unary::Unary##_Function<_ResultType, _Type, _NRank>, \
			UnaryOperators::_Function##Unary::UnaryContEmpty##_Function<_ResultType, _Type> \
		); \
	} \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace UnaryOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Negative(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			return -_Value;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Sqrt(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sqrt();
		else
			return static_cast<_Type>(std::sqrt(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		RSqrt(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.RSqrt();
		else
			return static_cast<_Type>(1 / std::sqrt(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Reciprocal(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Reciprocal();
		else
			return static_cast<_Type>(1 / _Value);
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Abs(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Abs(_mm256_set1_epi32(0x7FFFFFFF));
		else
			return static_cast<_Type>(std::abs(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Sin(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sin();
		else
			return static_cast<_Type>(std::sin(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Cos(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Cos();
		else
			return static_cast<_Type>(std::cos(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Tan(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Tan();
		else
			return static_cast<_Type>(std::tan(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ASin(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ASin();
		else
			return static_cast<_Type>(std::asin(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ACos(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ACos();
		else
			return static_cast<_Type>(std::acos(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ATan(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ATan();
		else
			return static_cast<_Type>(std::atan(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Sinh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sinh();
		else
			return static_cast<_Type>(std::sinh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Cosh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Cosh();
		else
			return static_cast<_Type>(std::cosh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Tanh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Tanh();
		else
			return static_cast<_Type>(std::tanh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ASinh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ASinh();
		else
			return static_cast<_Type>(std::asinh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ACosh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ACosh();
		else
			return static_cast<_Type>(std::acosh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		ATanh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ATanh();
		else
			return static_cast<_Type>(std::atanh(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Exp(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Exp();
		else
			return static_cast<_Type>(std::exp(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Log(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log();
		else
			return static_cast<_Type>(std::log(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Log2(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log2();
		else
			return static_cast<_Type>(std::log2(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Log10(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log10();
		else
			return static_cast<_Type>(std::log10(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Ceil(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Ceil();
		else
			return static_cast<_Type>(std::ceil(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Floor(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Floor();
		else
			return static_cast<_Type>(std::floor(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Round(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Round();
		else
			return static_cast<_Type>(std::round(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Trunc(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Trunc();
		else
			return static_cast<_Type>(std::trunc(_Value));
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<_Type> || IsArithmeticValue<_Type>, _Type>
		Frac(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Frac();
		else
			return static_cast<_Type>(static_cast<decltype(std::trunc(_Value))>(_Value) - std::trunc(_Value));
	}

}

_D_Dragonian_Lib_Operator_Unary_Function_Def(Negative, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sqrt, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(RSqrt, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Reciprocal, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Abs, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sin, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Cos, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Tan, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ASin, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ACos, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ATan, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sinh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Cosh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Tanh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ASinh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ACosh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ATanh, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Exp, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log2, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log10, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Ceil, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Floor, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Round, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Trunc, 8);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Frac, 8);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Unary_Function_Def