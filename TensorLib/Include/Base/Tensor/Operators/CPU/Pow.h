#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Cur_Operator(Arg1, Arg2) (PowImpl(Arg1, Arg2))

_D_Dragonian_Lib_Operator_Space_Begin

constexpr int64_t _D_Dragonian_Lib_Operator_Pow_Unfold = 8;

template <typename _Type>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
std::enable_if_t<
	(_Impl_Dragonian_Lib_Is_Floating_Point_v<_Type> || _Impl_Dragonian_Lib_Is_Complex_v<_Type>) ||
	(_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<Int64, _Type>) ||
	(_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32>) ||
	(_Impl_Dragonian_Lib_Could_Be_Converted_From_v<double, _Type> && _Impl_Dragonian_Lib_Could_Be_Converted_From_v<_Type, double>)
	, _Type> PowImpl(_Type _Val1, _Type _Val2)
{
	if constexpr (_Impl_Dragonian_Lib_Is_Floating_Point_v<_Type> || _Impl_Dragonian_Lib_Is_Complex_v<_Type>)
		return std::pow(_Val1, _Val2);
	else if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32>)
		return (_Type)powf((float)_Val1, (float)_Val2);
	else if constexpr (_Impl_Dragonian_Lib_Could_Be_Converted_From_v<double, _Type> && _Impl_Dragonian_Lib_Could_Be_Converted_From_v<_Type, double>)
		return (_Type)pow((double)_Val1, (double)_Val2);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename Type>
class _Impl_Dragonian_Lib_Has_Pow_Operator
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(_D_Dragonian_Lib_Cur_Operator(_Impl_Dragonian_Lib_Instance_Of<Objty>(), _Impl_Dragonian_Lib_Instance_Of<Objty>()), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _HasOperator = decltype(Check<Type>(0))::value;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Has_Pow_Operator_v = _Impl_Dragonian_Lib_Has_Pow_Operator<Type>::_HasOperator;

template<typename _Type>
void PowScalarCont(
	_Type* _Dest,
	const _Type* _Src,
	SizeType DestSize,
	std::shared_ptr<_Type> _ValPtr
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;
	const auto& _Value = *_ValPtr;

	SizeType i = 0;

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int32, Int64, Float32, Float64>)
	{
		auto _VectorizedValue1 = Vectorized<_Type>(_Value);
		auto _VectorizedValue2 = Vectorized<_Type>(_Value);
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Src1 = Vectorized<_Type>(_Src + i);
			auto _Src2 = Vectorized<_Type>(_Src + i + Stride);
			auto _Result1 = _Src1.Pow(_VectorizedValue1);
			auto _Result2 = _Src2.Pow(_VectorizedValue2);
			_Result1.Store(_Dest + i);
			_Result2.Store(_Dest + i + Stride);
		}
	}
	else
		for (; i < DestSize - OpThroughput; i += OpThroughput)
			for (int64_t j = 0; j < OpThroughput; ++j)
				_Dest[i + j] = _D_Dragonian_Lib_Cur_Operator(_Src[i + j], _Value);

	for (; i < DestSize; ++i)
		_Dest[i] = _D_Dragonian_Lib_Cur_Operator(_Src[i], _Value);
}

template<typename _Type, size_t _NRank>
void PowScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _Type* _Src,
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Pow_Operator_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld;
		
		const auto& _Value = *_ValPtr;

		const auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				_Dest[_IndexA] = _D_Dragonian_Lib_Cur_Operator(_Src[_IndexB], _Value);
			};
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
		const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();
		const SizeType* __restrict SrcViewStep = _SrcInfo.ViewStep.Data();
		const SizeType* __restrict SrcViewLeft = _SrcInfo.ViewLeft.Data();
		const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data();

		DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Pow_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplPowScalar(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const _Type& _Value,
	bool Continuous
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Pow_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
		ImplMultiThreadDouble(
			_Dest,
			_DestInfo,
			_Src,
			_SrcInfo,
			std::make_shared<_Type>(_Value),
			Continuous,
			PowScalar<_Type, _NRank>,
			PowScalarCont<_Type>
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void PowTensorCont(
	_Type* _Dest,
	const _Type* _Src1,
	const _Type* _Src2,
	SizeType DestSize,
	void*
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;

	SizeType i = 0;

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int32, Int64, Float32, Float64>)
	{
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Src11 = Vectorized<_Type>(_Src1 + i);
			auto _Src12 = Vectorized<_Type>(_Src1 + i + Stride);
			auto _Src21 = Vectorized<_Type>(_Src2 + i);
			auto _Src22 = Vectorized<_Type>(_Src2 + i + Stride);
			auto _Result1 = _Src11.Pow(_Src21);
			auto _Result2 = _Src12.Pow(_Src22);
			_Result1.Store(_Dest + i);
			_Result2.Store(_Dest + i + Stride);
		}
	}
	else
		for (; i < DestSize - OpThroughput; i += OpThroughput)
			for (int64_t j = 0; j < OpThroughput; ++j)
				_Dest[i + j] = _D_Dragonian_Lib_Cur_Operator(_Src1[i + j], _Src2[i + j]);

	for (; i < DestSize; ++i)
		_Dest[i] = _D_Dragonian_Lib_Cur_Operator(_Src1[i], _Src2[i]);
}

template <typename _Type, size_t _NRank>
void PowTensor(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _Type* _Src1,
	std::shared_ptr<OperatorParameter<_NRank>> _Src1InfoOld,
	const _Type* _Src2,
	std::shared_ptr<OperatorParameter<_NRank>> _Src2InfoOld,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Pow_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		const OperatorParameter<_NRank>& _Src1Info = *_Src1InfoOld;
		const OperatorParameter<_NRank>& _Src2Info = *_Src2InfoOld;
		

		const auto Func = [&](int64_t _IndexA, int64_t _IndexB, int64_t _IndexC)
			{
				_Dest[_IndexA] = _D_Dragonian_Lib_Cur_Operator(_Src1[_IndexB], _Src2[_IndexC]);
			};
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
		const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();
		const SizeType* __restrict Src1ViewStep = _Src1Info.ViewStep.Data();
		const SizeType* __restrict Src1ViewLeft = _Src1Info.ViewLeft.Data();
		const SizeType* __restrict Src1ViewStride = _Src1Info.ViewStride.Data();
		const SizeType* __restrict Src2ViewStep = _Src2Info.ViewStep.Data();
		const SizeType* __restrict Src2ViewLeft = _Src2Info.ViewLeft.Data();
		const SizeType* __restrict Src2ViewStride = _Src2Info.ViewStride.Data();

		TripleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Pow_Unfold>(
			0, 0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			Src1ViewStep, Src1ViewLeft, Src1ViewStride,
			Src2ViewStep, Src2ViewLeft, Src2ViewStride,
			Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplPowTensor(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src1,
	const OperatorParameter<_NRank>& _SrcInfo1,
	const _Type* _Src2,
	const OperatorParameter<_NRank>& _SrcInfo2,
	bool Continuous
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Pow_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
		ImplMultiThreadTriple(
			_Dest,
			_DestInfo,
			_Src1,
			_SrcInfo1,
			_Src2,
			_SrcInfo2,
			nullptr,
			Continuous,
			PowTensor<_Type, _NRank>,
			PowTensorCont<_Type>
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Cur_Operator