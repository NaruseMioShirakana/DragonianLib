#pragma once
#include "CPU.h"
_D_Dragonian_Lib_Operator_Space_Begin

constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Random_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Randn_Unfold = 8;

template<typename _TypeDest, typename _TypeSrc>
void AssignTensorCont(
	_TypeDest* _Dest,
	const _TypeSrc* _Src,
	SizeType DestSize,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc> && std::is_standard_layout_v<_TypeSrc>)
	{
		DestSize *= sizeof(_TypeDest);
		Vectorized<_TypeDest>::DragonianLibMemCpy(_Dest, _Src, DestSize);
	}
	else if constexpr (
		(_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_TypeDest, _TypeSrc> && std::is_move_assignable_v<_TypeDest>) ||
		(_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc> && std::is_copy_assignable_v<_TypeDest>)
		)
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc>)
					_Dest[i] = _Src[i];
				else
					_Dest[i] = _TypeDest(_Src[i]);
				++i;
			}
		}
		while (i < DestSize)
		{
			if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc>)
				_Dest[i] = _Src[i];
			else
				_Dest[i] = _TypeDest(_Src[i]);
			++i;
		}
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _TypeDest, typename _TypeSrc>
void AssignTensor(
	_TypeDest* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	const _TypeSrc* _Src,
	std::shared_ptr<OperatorParameter> _SrcInfoOld,
	void*
)
{
	if constexpr (
		(_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_TypeDest, _TypeSrc> && std::is_move_assignable_v<_TypeDest>) ||
		(_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc> && std::is_copy_assignable_v<_TypeDest>)
		)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		const OperatorParameter& _SrcInfo = *_SrcInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();

		const auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc>)
					_Dest[_IndexA] = _Src[_IndexB];
				else
					_Dest[_IndexA] = _TypeDest(_Src[_IndexB]);
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
			DoubleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 2)
			DoubleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 3)
			DoubleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 4)
			DoubleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 5)
			DoubleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 6)
			DoubleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 7)
			DoubleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 8)
			DoubleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 9)
			DoubleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStep, ViewLeft, ViewStride,
				SrcViewStep, SrcViewLeft, SrcViewStride,
				Func
			);
		else if (ViewRank == 10)
			DoubleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
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
template <typename _TypeSrc>
void OperatorsBase<_Type, Device::CPU>::ImplCast(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _TypeSrc* _Src,
	const OperatorParameter& _SrcInfo,
	bool Continuous
)
{
	if constexpr (_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_Type, _TypeSrc> && std::is_move_assignable_v<_Type>)
	{
		ImplMultiThreadDouble<_Type, _TypeSrc, void*>(
		   _Dest,
		   _DestInfo,
		   _Src,
		   _SrcInfo,
		   nullptr,
		   Continuous,
		   AssignTensor<_Type, _TypeSrc>,
		   AssignTensorCont<_Type, _TypeSrc>
	   );
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignTensor(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src,
	const OperatorParameter& _SrcInfo,
	bool Continuous
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		ImplMultiThreadDouble<_Type, _Type, void*>(
		   _Dest,
		   _DestInfo,
		   _Src,
		   _SrcInfo,
		   nullptr,
		   Continuous,
		   AssignTensor<_Type, _Type>,
		   AssignTensorCont<_Type, _Type>
	   );
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
struct _Struct_Buffer
{
	const _Type* _DestBegin;
	const _Type* _Src;
	SizeType _Count;
};

template<typename _Type>
void AssignBufferCont(
	_Type* _Dest,
	SizeType DestSize,
	_Struct_Buffer<_Type> _Value
)
{
	const auto Index = _Dest - _Value._DestBegin;
	if (Index >= _Value._Count) return;
	DestSize = std::min(DestSize, _Value._Count - Index);
	const auto _SrcPtr = _Value._Src + Index;
	if constexpr (std::is_standard_layout_v<_Type>)
		Vectorized<_Type>::DragonianLibMemCpy(_Dest, _SrcPtr, DestSize * sizeof(_Type));
	else if constexpr (std::is_copy_assignable_v<_Type>)
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				_Dest[i] = _SrcPtr[i];
				++i;
			}
		}
		while (i < DestSize)
		{
			_Dest[i] = _SrcPtr[i];
			++i;
		}
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignBuffer(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	_Struct_Buffer<_Type> _Value
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();

		const auto End = _Value._Src + _Value._Count;
		const _Type* _Src;
		if(std::ranges::count(_DestInfo.Begin, 0) == ViewRank)
			_Src = _Value._Src;
		else if (std::ranges::count(_DestInfo.Begin, 0) == ViewRank - 1)
		{
			SizeType TaskIndex = 1;
			for (SizeType i = 0; i < ViewRank; ++i)
				if (_DestInfo.Begin[i] != 0)
					TaskIndex *= _DestInfo.Begin[i];
				else
					TaskIndex *= _DestInfo.Shape[i];
			_Src = _Value._Src + TaskIndex;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
		if (_Src >= End) return;
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();

		if (ViewRank == 1)
			_D_Dragonian_Lib_Operator_Loop_S_0(
				_DestInfo,
				0,
				F, Begin[0], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis0FA] = *(_Src++);
				}
			);
		else if (ViewRank == 2)
			_D_Dragonian_Lib_Operator_Loop_S_1(
				_DestInfo,
				0,
				F, Begin[0], Begin[1], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis1FA] = *(_Src++);
				}
			);
		else if (ViewRank == 3)
			_D_Dragonian_Lib_Operator_Loop_S_2(
				_DestInfo,
				0,
				F, Begin[0], Begin[1], Begin[2], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis2FA] = *(_Src++);
				}
			);
		else if (ViewRank == 4)
			_D_Dragonian_Lib_Operator_Loop_S_3(
				_DestInfo,
				0,
				F, Begin[0], Begin[1], Begin[2], Begin[3], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis3FA] = *(_Src++);
				}
			);
		else if (ViewRank == 5)
			_D_Dragonian_Lib_Operator_Loop_S_4(
				_DestInfo,
				0,
				F, Begin[0], Begin[1], Begin[2], Begin[3], Begin[4], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis4FA] = *(_Src++);
				}
			);
		else if (ViewRank == 6)
			_D_Dragonian_Lib_Operator_Loop_S_5(
				_DestInfo,
				0,
				F, Begin[0], Begin[1], Begin[2], Begin[3], Begin[4], Begin[5], 0,
				{
					if (_Src >= End) return;
					_Dest[IndexAxis5FA] = *(_Src++);
				}
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignBuffer(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src,
	SizeType _Count,
	bool Continuous
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		_Struct_Buffer<_Type> _Value = { _Dest, _Src, _Count };
		ImplMultiThreadSingle<_Type, _Struct_Buffer<_Type>>(
			_Dest,
			_DestInfo,
			_Value,
			Continuous,
			AssignBuffer<_Type>,
			AssignBufferCont<_Type>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignScalarCont(
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

	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		if constexpr (_Impl_Dragonian_Lib_Is_Avx256_Supported_v<_Type>)
		{
			auto _VectorizedValue1 = Vectorized<_Type>(_Value);
			auto _VectorizedValue2 = Vectorized<_Type>(_Value);
			for (; i < DestSize - LoopStride; i += LoopStride)
			{
				_VectorizedValue1.Store(_Dest + i);
				_VectorizedValue2.Store(_Dest + i + Stride);
			}
		}
		else
			for (; i < DestSize - OpThroughput; i += OpThroughput)
				for (int64_t j = 0; j < OpThroughput; ++j)
					_Dest[i + j] = _Value;

		for (; i < DestSize; ++i)
			_Dest[i] = _Value;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		const OperatorParameter& _DestInfo = *_DestInfoOld;
		SizeType ViewRank = _DestInfo.GetRank();
		const auto& _Value = *_ValPtr;

		const auto Func = [&](int64_t _Index) { _Dest[_Index] = _Value; };
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
		const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

		if (ViewRank == 1)
			SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 2)
			SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 3)
			SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 4)
			SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 5)
			SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 6)
			SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 7)
			SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 8)
			SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 9)
			SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else if (ViewRank == 10)
			SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
				0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
			);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignScalar(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type& _Value,
	bool Continuous
)
{
	if constexpr (std::is_constructible_v<_Type>)
	{
		ImplMultiThreadSingle<_Type, std::shared_ptr<_Type>>(
		   _Dest,
		   _DestInfo,
		   std::make_shared<_Type>(_Value),
		   Continuous,
		   AssignScalar<_Type>,
		   AssignScalarCont<_Type>
	   );
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignRandnCont(
	_Type* _Dest,
	SizeType DestSize,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings._Mean, Settings._Sigma);

	SizeType i = 0;
	if constexpr (_Impl_Dragonian_Lib_Is_Complex_v<_Type>)
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 1] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 2] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 3] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 4] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 5] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 6] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 7] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
		}
		for (; i < DestSize; ++i)
			_Dest[i] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
	}
	else
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 1] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 2] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 3] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 4] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 5] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 6] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 7] = (_Type)NormalDistribution(RandomDevice);
		}
		for (; i < DestSize; ++i)
			_Dest[i] = (_Type)NormalDistribution(RandomDevice);
	}
}

template<typename _Type>
void AssignRandn(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings._Mean, Settings._Sigma);
	
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	const auto Func = [&](int64_t _Index)
		{
			if constexpr (_Impl_Dragonian_Lib_Is_Complex_v<_Type>)
				_Dest[_Index] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			else
				_Dest[_Index] = (_Type)NormalDistribution(RandomDevice);
		};
	if (ViewRank == 1)
		SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 2)
		SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 3)
		SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 4)
		SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 5)
		SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 6)
		SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 7)
		SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 8)
		SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 9)
		SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 10)
		SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRandn(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	double _Mean,
	double _Sigma,
	bool Continuous
)
{
	if constexpr (!_Impl_Dragonian_Lib_Is_Arithmetic_v<_Type>)
		_D_Dragonian_Lib_Not_Implemented_Error;
	else
	{
		using RandomType = _Impl_Dragonian_Lib_Random_Type<_Type>;
		using RandomNormalType = _Impl_Dragonian_Lib_Random_Normal_Type<_Type>;
		ImplMultiThreadSingle<_Type>(
			_Dest,
			_DestInfo,
			RandomSettings<_Type>{ (RandomType)0, (RandomType)0, (RandomNormalType)_Mean, (RandomNormalType)_Sigma},
			Continuous,
			AssignRandn<_Type>,
			AssignRandnCont<_Type>
		);
	}
}

template<typename _Type>
void AssignRandomCont(
	_Type* _Dest,
	SizeType DestSize,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings._Min, Settings._Max);

	SizeType i = 0;
	if constexpr (_Impl_Dragonian_Lib_Is_Complex_v<_Type>)
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 1] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 2] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 3] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 4] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 5] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 6] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 7] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
		}
		for (; i < DestSize; ++i)
			_Dest[i] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
	}
	else
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = (_Type)Distribution(RandomDevice);
			_Dest[i + 1] = (_Type)Distribution(RandomDevice);
			_Dest[i + 2] = (_Type)Distribution(RandomDevice);
			_Dest[i + 3] = (_Type)Distribution(RandomDevice);
			_Dest[i + 4] = (_Type)Distribution(RandomDevice);
			_Dest[i + 5] = (_Type)Distribution(RandomDevice);
			_Dest[i + 6] = (_Type)Distribution(RandomDevice);
			_Dest[i + 7] = (_Type)Distribution(RandomDevice);
		}
		for (;i < DestSize; ++i)
			_Dest[i] = (_Type)Distribution(RandomDevice);
	}
}

template<typename _Type>
void AssignRandom(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings._Min, Settings._Max);

	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();

	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	const auto Func = [&](int64_t _Index)
		{
			if constexpr (_Impl_Dragonian_Lib_Is_Complex_v<_Type>)
				_Dest[_Index] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			else
				_Dest[_Index] = (_Type)Distribution(RandomDevice);
		};

	if (ViewRank == 1)
		SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 2)
		SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 3)
		SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 4)
		SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 5)
		SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 6)
		SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 7)
		SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 8)
		SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 9)
		SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 10)
		SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRand(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type& _Min,
	const _Type& _Max,
	bool Continuous
)
{
	if constexpr (!_Impl_Dragonian_Lib_Is_Arithmetic_v<_Type>)
		_D_Dragonian_Lib_Not_Implemented_Error;
	else
	{
		using RandomType = _Impl_Dragonian_Lib_Random_Type<_Type>;
		using RandomNormalType = _Impl_Dragonian_Lib_Random_Normal_Type<_Type>;

		if constexpr (_Impl_Dragonian_Lib_Is_Complex_v<_Type>)
			ImplMultiThreadSingle<_Type>(
				_Dest,
				_DestInfo,
				RandomSettings<_Type>{
					(RandomType)_Min.real(), (RandomType)_Max.real(),
					(RandomNormalType)0., (RandomNormalType)0.
				},
				Continuous,
				AssignRandom<_Type>,
				AssignRandomCont<_Type>
			);
		else
			ImplMultiThreadSingle<_Type>(
				_Dest,
				_DestInfo,
				RandomSettings<_Type>{
					(RandomType)_Min, (RandomType)_Max,
					(RandomNormalType)0., (RandomNormalType)0.
				},
				Continuous,
				AssignRandom<_Type>,
				AssignRandomCont<_Type>
			);
	}
}

_D_Dragonian_Lib_Operator_Space_End