#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Cur_Operator(Arg1, Arg2) ((Arg1) + (Arg2))
#define _D_Dragonian_Lib_Cur_Inplace_Operator(Arg1, Arg2) ((Arg1) += (Arg2))

_D_Dragonian_Lib_Operator_Space_Begin

constexpr int64_t _D_Dragonian_Lib_Operator_Add_Unfold = 8;

template <typename Type>
class _Impl_Dragonian_Lib_Has_Add_Operator
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(_Impl_Dragonian_Lib_Instance_Of<Objty>()+ _Impl_Dragonian_Lib_Instance_Of<Objty>(), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _HasOperator = decltype(Check<Type>(0))::value || _Impl_Dragonian_Lib_Is_Arithmetic_v<Type>;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Has_Add_Operator_v = _Impl_Dragonian_Lib_Has_Add_Operator<Type>::_HasOperator;

template <typename Type>
class _Impl_Dragonian_Lib_Has_Add_Inplace_Operator
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(&Objty::operator+=(_Impl_Dragonian_Lib_Instance_Of<Objty>()), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _HasOperator = decltype(Check<Type>(0))::value || _Impl_Dragonian_Lib_Is_Arithmetic_v<Type>;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v = _Impl_Dragonian_Lib_Has_Add_Inplace_Operator<Type>::_HasOperator;

template<typename _Type>
void AddScalarInplaceCont(
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

template<typename _Type>
void AddScalarInplace(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();
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

		if (ViewRank == 1)
			SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 2)
			SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 3)
			SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 4)
			SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 5)
			SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 6)
			SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 7)
			SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 8)
			SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 9)
			SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 10)
			SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AddScalarCont(
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

template<typename _Type>
void AddScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	const _Type* _Src,
	std::shared_ptr<OperatorParameter> _SrcInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Add_Operator_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		const OperatorParameter& _SrcInfo = *_SrcInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();
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

		if (ViewRank == 1)
			DoubleTensorLoop<1, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 2)
			DoubleTensorLoop<2, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 3)
			DoubleTensorLoop<3, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 4)
			DoubleTensorLoop<4, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 5)
			DoubleTensorLoop<5, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 6)
			DoubleTensorLoop<6, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 7)
			DoubleTensorLoop<7, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 8)
			DoubleTensorLoop<8, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 9)
			DoubleTensorLoop<9, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 10)
			DoubleTensorLoop<10, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAddScalar(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src,
	const OperatorParameter& _SrcInfo,
	const _Type& _Value,
	bool Continuous
)
{
	if (_Dest == _Src)
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v<_Type>)
		{
			ImplMultiThreadSingle(
				_Dest,
				_DestInfo,
				std::make_shared<_Type>(_Value),
				Continuous,
				AddScalarInplace<_Type>,
				AddScalarInplaceCont<_Type>
			);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Add_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
			ImplMultiThreadDouble(
				_Dest,
				_DestInfo,
				_Src,
				_SrcInfo,
				std::make_shared<_Type>(_Value),
				Continuous,
				AddScalar<_Type>,
				AddScalarCont<_Type>
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
}

template<typename _Type>
void AddTensorInplaceCont(
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

template <typename _Type>
void AddTensorInplace(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	const _Type* _Src,
	std::shared_ptr<OperatorParameter> _SrcInfoOld,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		const OperatorParameter& _SrcInfo = *_SrcInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();

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

		if (ViewRank == 1)
			DoubleTensorLoop<1, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 2)
			DoubleTensorLoop<2, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 3)
			DoubleTensorLoop<3, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 4)
			DoubleTensorLoop<4, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 5)
			DoubleTensorLoop<5, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 6)
			DoubleTensorLoop<6, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 7)
			DoubleTensorLoop<7, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 8)
			DoubleTensorLoop<8, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 9)
			DoubleTensorLoop<9, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 10)
			DoubleTensorLoop<10, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AddTensorCont(
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

template <typename _Type>
void AddTensor(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	const _Type* _Src1,
	std::shared_ptr<OperatorParameter> _Src1InfoOld,
	const _Type* _Src2,
	std::shared_ptr<OperatorParameter> _Src2InfoOld,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Has_Add_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		const OperatorParameter& _Src1Info = *_Src1InfoOld;
		const OperatorParameter& _Src2Info = *_Src2InfoOld;
		SizeType ViewRank = _DestInfo.GetRank();

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

		if (ViewRank == 1)
			TripleTensorLoop<1, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 2)
			TripleTensorLoop<2, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 3)
			TripleTensorLoop<3, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 4)
			TripleTensorLoop<4, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 5)
			TripleTensorLoop<5, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 6)
			TripleTensorLoop<6, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 7)
			TripleTensorLoop<7, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 8)
			TripleTensorLoop<8, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 9)
			TripleTensorLoop<9, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else if (ViewRank == 10)
			TripleTensorLoop<10, _D_Dragonian_Lib_Operator_Add_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				Src1ViewStep, Src1ViewLeft, Src1ViewStride,
				Src2ViewStep, Src2ViewLeft, Src2ViewStride,
				Func
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAddTensor(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src1,
	const OperatorParameter& _SrcInfo1,
	const _Type* _Src2,
	const OperatorParameter& _SrcInfo2,
	bool Continuous
)
{
	if (_Dest == _Src1)
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v<_Type>)
		{
			ImplMultiThreadDouble(
				_Dest,
				_DestInfo,
				_Src2,
				_SrcInfo2,
				nullptr,
				Continuous,
				AddTensorInplace<_Type>,
				AddTensorInplaceCont<_Type>
			);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
	{
		if constexpr (_Impl_Dragonian_Lib_Has_Add_Operator_v<_Type> && std::is_move_assignable_v<_Type>)
			ImplMultiThreadTriple(
				_Dest,
				_DestInfo,
				_Src1,
				_SrcInfo1,
				_Src2,
				_SrcInfo2,
				nullptr,
				Continuous,
				AddTensor<_Type>,
				AddTensorCont<_Type>
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
}

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Cur_Operator
#undef _D_Dragonian_Lib_Cur_Inplace_Operator