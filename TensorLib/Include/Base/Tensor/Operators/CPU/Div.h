#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Cur_Operator(Arg1, Arg2) ((Arg1) / (Arg2))
#define _D_Dragonian_Lib_Cur_Inplace_Operator(Arg1, Arg2) ((Arg1) /= (Arg2))

_D_Dragonian_Lib_Operator_Space_Begin

constexpr int64_t _D_Dragonian_Lib_Operator_Div_Unfold = 8;

template <typename Type>
class _Impl_Dragonian_Lib_Has_Div_Operator
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(_Impl_Dragonian_Lib_Instance_Of<Objty>() / _Impl_Dragonian_Lib_Instance_Of<Objty>(), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _HasOperator = decltype(Check<Type>(0))::value;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Has_Div_Operator_v = _Impl_Dragonian_Lib_Has_Div_Operator<Type>::_HasOperator;

template <typename Type>
class _Impl_Dragonian_Lib_Has_Div_Inplace_Operator
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(&Objty::operator/=(_Impl_Dragonian_Lib_Instance_Of<Objty>()), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _HasOperator = decltype(Check<Type>(0))::value || _Impl_Dragonian_Lib_Is_Arithmetic_v<Type>;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v = _Impl_Dragonian_Lib_Has_Div_Inplace_Operator<Type>::_HasOperator;

template<typename _Type>
void DivScalarInplaceCont(
	_Type* _Dest,
	SizeType DestSize,
	std::shared_ptr<_Type> _ValPtr
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;
	const auto& _Value = *_ValPtr;

	SizeType i = 0;

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32, Int64, Float32, Float64>)
	{
		auto _VectorizedValue1 = Vectorized<_Type>(_Value);
		auto _VectorizedValue2 = Vectorized<_Type>(_Value);
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Dest1 = Vectorized<_Type>(_Dest + i);
			auto _Dest2 = Vectorized<_Type>(_Dest + i + Stride);
			auto _Result1 = _D_Dragonian_Lib_Cur_Operator(_Dest1, _VectorizedValue1);
			auto _Result2 = _D_Dragonian_Lib_Cur_Operator(_Dest2, _VectorizedValue2);
			_Result1.Store(_Dest + i);
			_Result2.Store(_Dest + i + Stride);
		}
	}
	else
		for (; i < DestSize - OpThroughput; i += OpThroughput)
			for (int64_t j = 0; j < OpThroughput; ++j)
				_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[i + j], _Value);

	for (; i < DestSize; ++i)
		_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[i], _Value);
}

template<typename _Type, size_t _NRank>
void DivScalarInplace(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		
		const auto& _Value = *_ValPtr;

		const auto Func = [&](int64_t _Index)
			{
				_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[_Index], _Value);
			};
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
		const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

		SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Div_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void DivScalarCont(
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

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32, Int64, Float32, Float64>)
	{
		auto _VectorizedValue1 = Vectorized<_Type>(_Value);
		auto _VectorizedValue2 = Vectorized<_Type>(_Value);
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Src1 = Vectorized<_Type>(_Src + i);
			auto _Src2 = Vectorized<_Type>(_Src + i + Stride);
			auto _Result1 = _D_Dragonian_Lib_Cur_Operator(_Src1, _VectorizedValue1);
			auto _Result2 = _D_Dragonian_Lib_Cur_Operator(_Src2, _VectorizedValue2);
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
void DivScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _Type* _Src,
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Div_Operator_v<_Type>)
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

		DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Div_Unfold>(
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
void OperatorsBase<_Type, Device::CPU>::ImplDivScalar(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const _Type& _Value,
	bool Continuous
)
{
	if (_Dest == _Src)
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v<_Type>)
		{
			ImplMultiThreadSingle(
				_Dest,
				_DestInfo,
				std::make_shared<_Type>(_Value),
				Continuous,
				DivScalarInplace<_Type, _NRank>,
				DivScalarInplaceCont<_Type>
			);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Div_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
			ImplMultiThreadDouble(
				_Dest,
				_DestInfo,
				_Src,
				_SrcInfo,
				std::make_shared<_Type>(_Value),
				Continuous,
				DivScalar<_Type, _NRank>,
				DivScalarCont<_Type>
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
}

template<typename _Type>
void DivTensorInplaceCont(
	_Type* _Dest,
	const _Type* _Src,
	SizeType DestSize,
	void*
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;

	SizeType i = 0;

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32, Int64, Float32, Float64>)
	{
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Dest1 = Vectorized<_Type>(_Dest + i);
			auto _Dest2 = Vectorized<_Type>(_Dest + i + Stride);
			auto _Src1 = Vectorized<_Type>(_Src + i);
			auto _Src2 = Vectorized<_Type>(_Src + i + Stride);
			auto _Result1 = _D_Dragonian_Lib_Cur_Operator(_Dest1, _Src1);
			auto _Result2 = _D_Dragonian_Lib_Cur_Operator(_Dest2, _Src2);
			_Result1.Store(_Dest + i);
			_Result2.Store(_Dest + i + Stride);
		}
	}
	else
		for (; i < DestSize - OpThroughput; i += OpThroughput)
			for (int64_t j = 0; j < OpThroughput; ++j)
				_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[i + j], _Src[i + j]);

	for (; i < DestSize; ++i)
		_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[i], _Src[i]);
}

template <typename _Type, size_t _NRank>
void DivTensorInplace(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _Type* _Src,
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld;
		

		const auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				_D_Dragonian_Lib_Cur_Inplace_Operator(_Dest[_IndexA], _Src[_IndexB]);
			};
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
		const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();
		const SizeType* __restrict SrcViewStep = _SrcInfo.ViewStep.Data();
		const SizeType* __restrict SrcViewLeft = _SrcInfo.ViewLeft.Data();
		const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data();

		DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Div_Unfold>(
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

template<typename _Type>
void DivTensorCont(
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

	if constexpr (_Impl_Dragonian_Lib_Is_Any_Of_v<_Type, Int8, Int16, Int32, Int64, Float32, Float64>)
	{
		for (; i < DestSize - LoopStride; i += LoopStride)
		{
			auto _Src11 = Vectorized<_Type>(_Src1 + i);
			auto _Src12 = Vectorized<_Type>(_Src1 + i + Stride);
			auto _Src21 = Vectorized<_Type>(_Src2 + i);
			auto _Src22 = Vectorized<_Type>(_Src2 + i + Stride);
			auto _Result1 = _D_Dragonian_Lib_Cur_Operator(_Src11, _Src21);
			auto _Result2 = _D_Dragonian_Lib_Cur_Operator(_Src12, _Src22);
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
void DivTensor(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _Type* _Src1,
	std::shared_ptr<OperatorParameter<_NRank>> _Src1InfoOld,
	const _Type* _Src2,
	std::shared_ptr<OperatorParameter<_NRank>> _Src2InfoOld,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Div_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
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

		TripleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Div_Unfold>(
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
void OperatorsBase<_Type, Device::CPU>::ImplDivTensor(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src1,
	const OperatorParameter<_NRank>& _SrcInfo1,
	const _Type* _Src2,
	const OperatorParameter<_NRank>& _SrcInfo2,
	bool Continuous
)
{
	if (_Dest == _Src1)
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v<_Type>)
		{
			ImplMultiThreadDouble(
				_Dest,
				_DestInfo,
				_Src2,
				_SrcInfo2,
				nullptr,
				Continuous,
				DivTensorInplace<_Type, _NRank>,
				DivTensorInplaceCont<_Type>
			);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Div_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
			ImplMultiThreadTriple(
				_Dest,
				_DestInfo,
				_Src1,
				_SrcInfo1,
				_Src2,
				_SrcInfo2,
				nullptr,
				Continuous,
				DivTensor<_Type, _NRank>,
				DivTensorCont<_Type>
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
}

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Cur_Operator
#undef _D_Dragonian_Lib_Cur_Inplace_Operator