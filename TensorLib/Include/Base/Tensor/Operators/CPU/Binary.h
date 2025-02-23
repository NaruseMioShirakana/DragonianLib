#pragma once
#include "CPU.h"
#include "Libraries/Util/Logger.h"

#define _D_Dragonian_Lib_Operator_Binary_Function_Def(_Function, Unfold) namespace BinaryOperators { namespace _Function##Binary { \
 \
constexpr int64_t _D_Dragonian_Lib_Operator_Binary_Unfold = 8; \
 \
template <class _ValueType> \
concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) { _D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l); }; \
template <class _ValueType> \
concept HasInplaceOperatorValue = requires(_ValueType & __r, _ValueType & __l) { _D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function##Inplace(__r, __l); }; \
 \
template<typename _Type> \
void BinaryScalarInplaceCont( \
	_Type* _Dest, \
	SizeType DestSize, \
	std::shared_ptr<_Type> _ValPtr \
) \
{ \
	constexpr int64_t OpThroughput = 2; \
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type)); \
	constexpr int64_t LoopStride = OpThroughput * Stride; \
	const auto& _Value = *_ValPtr; \
 \
	SizeType i = 0; \
 \
	bool EnableAvx = IsAvx256SupportedValue<_Type>; \
	try \
	{ \
		if constexpr (IsAvx256SupportedValue<_Type>) \
			_Function(Vectorized<_Type>(_Dest + i), Vectorized<_Type>(_Dest + i)); \
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
		{ \
			auto _VectorizedValue1 = Vectorized<_Type>(_Value); \
			auto _VectorizedValue2 = Vectorized<_Type>(_Value); \
			for (; i < DestSize - LoopStride; i += LoopStride) \
			{ \
				auto _Dest1 = Vectorized<_Type>(_Dest + i); \
				auto _Dest2 = Vectorized<_Type>(_Dest + i + Stride); \
				auto _Result1 = _Function(_Dest1, _VectorizedValue1); \
				auto _Result2 = _Function(_Dest2, _VectorizedValue2); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Function##Inplace(_Dest[i + j], _Value); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Function##Inplace(_Dest[i + j], _Value); \
 \
	for (; i < DestSize; ++i) \
		_Function##Inplace(_Dest[i], _Value); \
} \
 \
template<typename _Type, size_t _NRank> \
void BinaryScalarInplace( \
	_Type* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	std::shared_ptr<_Type> _ValPtr \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
	const auto& _Value = *_ValPtr; \
 \
	const auto Func = [&](int64_t _Index) \
		{ \
			_Function##Inplace(_Dest[_Index], _Value); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
 \
	SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Binary_Unfold>( \
		0, Shape, Begin, ViewStride, Func \
	); \
} \
 \
template<typename _Type> \
void BinaryScalarCont( \
	_Type* _Dest, \
	const _Type* _Src, \
	SizeType DestSize, \
	std::shared_ptr<_Type> _ValPtr \
) \
{ \
	constexpr int64_t OpThroughput = 2; \
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type)); \
	constexpr int64_t LoopStride = OpThroughput * Stride; \
	const auto& _Value = *_ValPtr; \
 \
	SizeType i = 0; \
 \
	bool EnableAvx = IsAvx256SupportedValue<_Type>; \
	try \
	{ \
		if constexpr (IsAvx256SupportedValue<_Type>) \
			_Function(Vectorized<_Type>(_Dest + i), Vectorized<_Type>(_Dest + i)); \
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
		{ \
			auto _VectorizedValue1 = Vectorized<_Type>(_Value); \
			auto _VectorizedValue2 = Vectorized<_Type>(_Value); \
			for (; i < DestSize - LoopStride; i += LoopStride) \
			{ \
				auto _Src1 = Vectorized<_Type>(_Src + i); \
				auto _Src2 = Vectorized<_Type>(_Src + i + Stride); \
				auto _Result1 = _Function(_Src1, _VectorizedValue1); \
				auto _Result2 = _Function(_Src2, _VectorizedValue2); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Dest[i + j] = _Function(_Src[i + j], _Value); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Dest[i + j] = _Function(_Src[i + j], _Value); \
 \
	for (; i < DestSize; ++i) \
		_Dest[i] = _Function(_Src[i], _Value); \
} \
 \
template<typename _Type, size_t _NRank> \
void BinaryScalar( \
	_Type* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	const _Type* _Src, \
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld, \
	std::shared_ptr<_Type> _ValPtr \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
	const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld; \
	const auto& _Value = *_ValPtr; \
 \
	const auto Func = [&](int64_t _IndexA, int64_t _IndexB) \
		{ \
			_Dest[_IndexA] = _Function(_Src[_IndexB], _Value); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
	const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data(); \
 \
	DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Binary_Unfold>( \
		0, 0, \
		Shape, Begin, \
		ViewStride, SrcViewStride, \
		Func \
	); \
} \
 \
template<typename _Type> \
void BinaryTensorInplaceCont( \
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
			_Function(Vectorized<_Type>(_Dest + i), Vectorized<_Type>(_Dest + i)); \
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
				auto _Src1 = Vectorized<_Type>(_Src + i); \
				auto _Src2 = Vectorized<_Type>(_Src + i + Stride); \
				auto _Result1 = _Function(_Dest1, _Src1); \
				auto _Result2 = _Function(_Dest2, _Src2); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Function##Inplace(_Dest[i + j], _Src[i + j]); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Function##Inplace(_Dest[i + j], _Src[i + j]); \
 \
	for (; i < DestSize; ++i) \
		_Function##Inplace(_Dest[i], _Src[i]); \
} \
 \
template <typename _Type, size_t _NRank> \
void BinaryTensorInplace( \
	_Type* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	const _Type* _Src, \
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld, \
	void* \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
	const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld; \
 \
	const auto Func = [&](int64_t _IndexA, int64_t _IndexB) \
		{ \
			_Function##Inplace(_Dest[_IndexA], _Src[_IndexB]); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
	const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data(); \
 \
	DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Binary_Unfold>( \
		0, 0, \
		Shape, Begin, \
		ViewStride, SrcViewStride, \
		Func \
	); \
} \
 \
template<typename _Type> \
void BinaryTensorCont( \
	_Type* _Dest, \
	const _Type* _Src1, \
	const _Type* _Src2, \
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
			_Function(Vectorized<_Type>(_Dest + i), Vectorized<_Type>(_Dest + i)); \
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
				auto _Src11 = Vectorized<_Type>(_Src1 + i); \
				auto _Src12 = Vectorized<_Type>(_Src1 + i + Stride); \
				auto _Src21 = Vectorized<_Type>(_Src2 + i); \
				auto _Src22 = Vectorized<_Type>(_Src2 + i + Stride); \
				auto _Result1 = _Function(_Src11, _Src21); \
				auto _Result2 = _Function(_Src12, _Src22); \
				_Result1.Store(_Dest + i); \
				_Result2.Store(_Dest + i + Stride); \
			} \
		else \
			for (; i < DestSize - OpThroughput; i += OpThroughput) \
				for (int64_t j = 0; j < OpThroughput; ++j) \
					_Dest[i + j] = _Function(_Src1[i + j], _Src2[i + j]); \
	} \
	else \
		for (; i < DestSize - OpThroughput; i += OpThroughput) \
			for (int64_t j = 0; j < OpThroughput; ++j) \
				_Dest[i + j] = _Function(_Src1[i + j], _Src2[i + j]); \
 \
	for (; i < DestSize; ++i) \
		_Dest[i] = _Function(_Src1[i], _Src2[i]); \
} \
 \
template <typename _Type, size_t _NRank> \
void BinaryTensor( \
	_Type* _Dest, \
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld, \
	const _Type* _Src1, \
	std::shared_ptr<OperatorParameter<_NRank>> _Src1InfoOld, \
	const _Type* _Src2, \
	std::shared_ptr<OperatorParameter<_NRank>> _Src2InfoOld, \
	void* \
) \
{ \
	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld; \
	const OperatorParameter<_NRank>& _Src1Info = *_Src1InfoOld; \
	const OperatorParameter<_NRank>& _Src2Info = *_Src2InfoOld; \
 \
	const auto Func = [&](int64_t _IndexA, int64_t _IndexB, int64_t _IndexC) \
		{ \
			_Dest[_IndexA] = _Function(_Src1[_IndexB], _Src2[_IndexC]); \
		}; \
	const SizeType* __restrict Shape = _DestInfo.Shape.Data(); \
	const SizeType* __restrict Begin = _DestInfo.Begin.Data(); \
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data(); \
	const SizeType* __restrict Src1ViewStride = _Src1Info.ViewStride.Data(); \
	const SizeType* __restrict Src2ViewStride = _Src2Info.ViewStride.Data(); \
 \
	TripleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Binary_Unfold>( \
		0, 0, 0, \
		Shape, Begin, \
		ViewStride, Src1ViewStride, Src2ViewStride, \
		Func \
	); \
} \
} \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Scalar( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
) \
{ \
	try \
	{ \
		_Type Test = BinaryOperators::_Function(*_Src, _Value); \
	} \
	catch (std::exception& e) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(e.what()); \
	} \
	if constexpr (!BinaryOperators::_Function##Binary::HasOperatorValue<_Type>) \
		_D_Dragonian_Lib_Not_Implemented_Error; \
	if constexpr (BinaryOperators::_Function##Binary::HasInplaceOperatorValue<_Type>) \
	{ \
		if (_Dest == _Src) \
		{ \
			ImplMultiThreadSingle( \
				_Dest, \
				_DestInfo, \
				std::make_shared<_Type>(_Value), \
				Continuous, \
				BinaryOperators::_Function##Binary::BinaryScalarInplace<_Type, _NRank>, \
				BinaryOperators::_Function##Binary::BinaryScalarInplaceCont<_Type> \
			); \
			return; \
		} \
	} \
	ImplMultiThreadDouble( \
		_Dest, \
		_DestInfo, \
		_Src, \
		_SrcInfo, \
		std::make_shared<_Type>(_Value), \
		Continuous, \
		BinaryOperators::_Function##Binary::BinaryScalar<_Type, _NRank>, \
		BinaryOperators::_Function##Binary::BinaryScalarCont<_Type> \
	); \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Tensor( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src1, \
	const OperatorParameter<_NRank>& _SrcInfo1, \
	const _Type* _Src2, \
	const OperatorParameter<_NRank>& _SrcInfo2, \
	bool Continuous \
) \
{ \
	try \
	{ \
		_Type Test = BinaryOperators::_Function(*_Src1, *_Src2); \
	} \
	catch (std::exception& e) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(e.what()); \
	} \
	if constexpr (!BinaryOperators::_Function##Binary::HasOperatorValue<_Type>) \
		_D_Dragonian_Lib_Not_Implemented_Error; \
	if constexpr (BinaryOperators::_Function##Binary::HasInplaceOperatorValue<_Type>) \
	{ \
		if (_Dest == _Src1) \
		{ \
			ImplMultiThreadDouble( \
				_Dest, \
				_DestInfo, \
				_Src2, \
				_SrcInfo2, \
				nullptr, \
				Continuous, \
				BinaryOperators::_Function##Binary::BinaryTensorInplace<_Type, _NRank>, \
				BinaryOperators::_Function##Binary::BinaryTensorInplaceCont<_Type> \
			); \
			return; \
		} \
	} \
	ImplMultiThreadTriple( \
		_Dest, \
		_DestInfo, \
		_Src1, \
		_SrcInfo1, \
		_Src2, \
		_SrcInfo2, \
		nullptr, \
		Continuous, \
		BinaryOperators::_Function##Binary::BinaryTensor<_Type, _NRank>, \
		BinaryOperators::_Function##Binary::BinaryTensorCont<_Type> \
	); \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace BinaryOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { {_Left + _Right}->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Add(const _Type& _Left, const _Type& _Right)
	{
		return _Left + _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left += _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		AddInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left += _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left - _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sub(const _Type& _Left, const _Type& _Right)
	{
		return _Left - _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left -= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		SubInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left -= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left * _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mul(const _Type& _Left, const _Type& _Right)
	{
		return _Left * _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left *= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		MulInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left *= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left / _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Div(const _Type& _Left, const _Type& _Right)
	{
		return _Left / _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left /= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		DivInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left /= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { std::fmod(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left, _Type & _Right) { { _Left% _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mod(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (IsFloatingPointValue<_Type>)
			return std::fmod(_Left, _Right);
		else
			return _Left % _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left = std::fmod(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; } ||
		requires(_Type & _Left, _Type & _Right) { { _Left %= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ModInplace(_Type& _Left, const _Type& _Right)
	{
		if constexpr (IsFloatingPointValue<_Type>)
			return _Left = std::fmod(_Left, _Right);
		else
			return _Left %= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left && _Right }; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		And(const _Type& _Left, const _Type& _Right)
	{
		return _Type(_Left && _Right);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left = _Left && _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		AndInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left = _Left && _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left || _Right }; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Or(const _Type& _Left, const _Type& _Right)
	{
		return _Type(_Left || _Right);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left = _Left || _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		OrInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left = _Left || _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left & _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryAnd(_Type& _Left, const _Type& _Right)
	{
		return _Left & _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left &= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
	BinaryAndInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left &= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left | _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryOr(_Type& _Left, const _Type& _Right)
	{
		return _Left | _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left |= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
	BinaryOrInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left |= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left ^ _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Xor(const _Type& _Left, const _Type& _Right)
	{
		return _Left ^ _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left ^= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		XorInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left ^= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left << _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LShift(const _Type& _Left, const _Type& _Right)
	{
		return _Left << _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left <<= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LShiftInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left <<= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		(requires(_Type& _Left, _Type& _Right) { { _Left >> _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; })
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		RShift(const _Type& _Left, const _Type& _Right)
	{
		return _Left >> _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left >>= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
	RShiftInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left >>= _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left.Pow(_Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type& _Left, _Type& _Right) { { std::pow(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Pow(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Left.Pow(_Right);
		else
			return std::pow(_Left, _Right);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left = Pow(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type&>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		PowInplace(_Type& _Left, const _Type& _Right)
	{
		return _Left = Pow(_Left, _Right);
	}
}

_D_Dragonian_Lib_Operator_Binary_Function_Def(Add, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Sub, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mul, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Div, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mod, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(And, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Or, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Xor, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BinaryOr, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BinaryAnd, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(LShift, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(RShift, 8);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Pow, 8);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Function_Def