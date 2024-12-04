#pragma once
#include "CPU.h"
#include "Libraries/Util/Logger.h"

#define _D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(_Function, Unfold) namespace ComparisonOperators { namespace _Function##Binary { \
 \
constexpr int64_t _D_Dragonian_Lib_Operator_Binary_Unfold = 8; \
template <typename Type> \
constexpr bool HasOperatorValue = decltype(TypeTraits::IsInvokableWith::CheckConst(_Function##<Type>, InstanceOf<Type>(), InstanceOf<Type>()))::value; \
 \
template<typename _Type> \
void BinaryScalarCont( \
	bool* _Dest, \
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
			_Function(Vectorized<_Type>(_Src), Vectorized<_Type>(_Src)); \
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
				_Result1.StoreBool(_Dest + i); \
				_Result2.StoreBool(_Dest + i + Stride); \
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
	bool* _Dest, \
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
void BinaryTensorCont( \
	bool* _Dest, \
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
			_Function(Vectorized<_Type>(_Src1), Vectorized<_Type>(_Src1)); \
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
				_Result1.StoreBool(_Dest + i); \
				_Result2.StoreBool(_Dest + i + Stride); \
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
	bool* _Dest, \
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
	bool* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
) \
{ \
	try \
	{ \
		bool Test = ComparisonOperators::_Function(*_Src, _Value); \
	} \
	catch (std::exception& e) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(e.what()); \
	} \
	if constexpr (!ComparisonOperators::_Function##Binary::HasOperatorValue<_Type>) \
		_D_Dragonian_Lib_Not_Implemented_Error; \
	ImplMultiThreadDouble( \
		_Dest, \
		_DestInfo, \
		_Src, \
		_SrcInfo, \
		std::make_shared<_Type>(_Value), \
		Continuous, \
		ComparisonOperators::_Function##Binary::BinaryScalar<_Type, _NRank>, \
		ComparisonOperators::_Function##Binary::BinaryScalarCont<_Type> \
	); \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Tensor( \
	bool* _Dest, \
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
		bool Test = ComparisonOperators::_Function(*_Src1, *_Src2); \
	} \
	catch (std::exception& e) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(e.what()); \
	} \
	if constexpr (!ComparisonOperators::_Function##Binary::HasOperatorValue<_Type>) \
		_D_Dragonian_Lib_Not_Implemented_Error; \
	ImplMultiThreadTriple( \
		_Dest, \
		_DestInfo, \
		_Src1, \
		_SrcInfo1, \
		_Src2, \
		_SrcInfo2, \
		nullptr, \
		Continuous, \
		ComparisonOperators::_Function##Binary::BinaryTensor<_Type, _NRank>, \
		ComparisonOperators::_Function##Binary::BinaryTensorCont<_Type> \
	); \
} 

_D_Dragonian_Lib_Operator_Space_Begin

namespace ComparisonOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		Equal(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) <= std::numeric_limits<Type>::epsilon();
		else
			return A == B;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		NotEqual(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) > std::numeric_limits<Type>::epsilon();
		else
			return A != B;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		Greater(const Type& A, const Type& B)
	{
		return A > B;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		GreaterEqual(const Type& A, const Type& B)
	{
		return A >= B;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		Less(const Type& A, const Type& B)
	{
		return A < B;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsVectorizedValue<Type> || IsArithmeticValue<Type>,ConditionalType<IsVectorizedValue<Type>, Type, bool>>
		LessEqual(const Type& A, const Type& B)
	{
		return A <= B;
	}
}

_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Equal, 0)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(NotEqual, 0)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Greater, 0)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(GreaterEqual, 0)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Less, 0)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(LessEqual, 0)

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Bool_Function_Def