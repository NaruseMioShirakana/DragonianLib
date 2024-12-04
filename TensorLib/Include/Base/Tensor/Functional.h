﻿#pragma once
#include "Tensor.h"
#include "Libraries/NumpySupport/NumpyFileFormat.h"
#include <ostream>

_D_Dragonian_Lib_Space_Begin

template <typename ..._TIndices>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) IDim(_TIndices&& ..._Indices)
{
	return Dimensions<sizeof...(_TIndices)>({ std::forward<_TIndices>(_Indices)... });
}

template <typename ..._ValueTypes>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) IArray(_ValueTypes&& ..._Values)
{
	return IDLArray<TypeTraits::GetVaListTypeAtType<0, _ValueTypes...>, sizeof...(_ValueTypes)>({ std::forward<_ValueTypes>(_Values)... });
}

template <typename ..._ArgTypes>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) IRanges(_ArgTypes&& ..._Args)
{
	return VRanges<sizeof...(_ArgTypes)>({ std::forward<_ArgTypes>(_Args)... });
}

namespace Functional
{
	template <typename _MyValueType, size_t _NRank, Device _MyDevice>
	void NumpySave(
		const std::wstring& _Path,
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor
	)
	{
		NumpyFileFormat::SaveNumpyFile(_Path, _Tensor.Shape(), _Tensor.Data(), _Tensor.TotalSize());
	}

	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	Tensor<_MyValueType, _NRank, _MyDevice> NumpyLoad(const std::wstring& _Path)
	{
		auto [VecShape, VecData] = NumpyFileFormat::LoadNumpyFile(_Path);

		Dimensions<_NRank> Shape;
		if (VecShape.Size() != _NRank)
			_D_Dragonian_Lib_Throw_Exception("The rank of the tensor is not compatible with the numpy file.");
		Shape.Assign(VecShape.Data());
		auto Alloc = VecData.GetAllocator();
		auto Ret = VecData.Release();
		Ret.second /= sizeof(_MyValueType);
		if (Ret.second != static_cast<size_t>(Shape.Multiply()))
			_D_Dragonian_Lib_Throw_Exception("The data size of the tensor is not compatible with the numpy file.");
		return Tensor<_MyValueType, _NRank, _MyDevice>::FromBuffer(Shape, (_MyValueType*)Ret.first, Ret.second, Alloc);
	}

	/**
	 * @brief Create a new tensor with the specified shape.
	 * @param MyShape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		std::is_trivial_v<_MyValueType> ||
		std::is_constructible_v<_MyValueType>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Empty(
			const Dimensions<_NRank>& MyShape
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::New(MyShape);
	}

	/**
	 * @brief Create a new tensor with the specified shape and initialize it with the specified args.
	 * @param MyShape The shape of the tensor.
	 * @param Arg0 The first value.
	 * @param Args The rest values.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU, typename _First, typename ...Rest>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		std::is_constructible_v<_MyValueType, _First, Rest...>,
		Tensor<_MyValueType, _NRank, _MyDevice>> ConstructTensor(
			const Dimensions<_NRank>& MyShape,
			_First Arg0,
			Rest ...Args
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::New(MyShape, Arg0, Args...);
	}

	/**
	 * @brief Create an empty new tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		std::is_trivial_v<_MyValueType>,
		Tensor<_MyValueType, _NRank, _MyDevice>> EmptyTensor()
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::New();
	}

	/**
	 * @brief Create a new scalar tensor with the specified value.
	 * @param _Val The value of the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		std::is_constructible_v<_MyValueType, _MyValueType>,
		Tensor<_MyValueType, 1, _MyDevice>> NewScalar(
			const _MyValueType& _Val
		)
	{
		return Tensor<_MyValueType, 1, _MyDevice>::New({ 1 }, _Val);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with ones.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, decltype(1)>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Ones(
			const Dimensions<_NRank>& _Shape
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::Ones(_Shape);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with zeros.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, decltype(0)>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Zeros(
			const Dimensions<_NRank>& _Shape
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::Zeros(_Shape);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with a constant value.
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> ConstantOf(
			const Dimensions<_NRank>& _Shape,
			const _MyValueType& _Val
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::ConstantOf(_Shape, _Val);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::IsArithmeticValue<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Rand(
			const Dimensions<_NRank>& _Shape,
			const _MyValueType& Min,
			const _MyValueType& Max
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::Rand(_Shape, Min, Max);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::IsArithmeticValue<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Randn(
			const Dimensions<_NRank>& _Shape,
			double _Mean = 0.,
			double _Sigma = 1.
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::Randn(_Shape, _Mean, _Sigma);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with ones.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, decltype(1)>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> OnesLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference
		)
	{
		return Ones<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with zeros.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, decltype(0)>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> ZerosLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference
		)
	{
		return Zeros<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with a constant value.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> ConstantLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference,
			const _MyValueType& _Val
		)
	{
		return ConstantOf<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape(), _Val);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::IsArithmeticValue<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> RandLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference,
			const _MyValueType& Min,
			const _MyValueType& Max
		)
	{
		return Rand<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape(), Min, Max);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::IsArithmeticValue<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> RandnLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference,
			double _Mean = 0.,
			double _Sigma = 1.
		)
	{
		return Randn<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape(), _Mean, _Sigma);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		std::is_trivial_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> EmptyLike(
			const Tensor<_MyValueType, _NRank, _MyDevice>& _ShapeReference
		)
	{
		return Empty<_MyValueType, _NRank, _MyDevice>(_ShapeReference.Shape());
	}

	template <typename _MyValueType = Float32, size_t _NRank = 2, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		Operators::BinaryOperators::AddBinary::HasOperatorValue<_MyValueType>&&
		Operators::BinaryOperators::MulBinary::HasOperatorValue<_MyValueType>&&
		std::is_move_assignable_v<_MyValueType>&&
		std::is_constructible_v<_MyValueType>>,
		Tensor<_MyValueType, _NRank, _MyDevice>> Arange(
			_MyValueType _Begin,
			_MyValueType _End,
			_MyValueType _Step
		)
	{
		return Tensor<_MyValueType, _NRank, _MyDevice>::Arange(_Begin, _End, _Step);
	}

	template <typename _MyValueType = Float32, Device _MyDevice = Device::CPU>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		std::enable_if_t<
		TypeTraits::AndValue<
		TypeTraits::IsSameTypeValue<_MyValueType, _MyValueType>,
		TypeTraits::CouldBeConvertedFromValue<_MyValueType, _MyValueType>&&
		std::is_copy_assignable_v<_MyValueType>>,
		Tensor<_MyValueType, 1, _MyDevice>> FromBuffer(const _MyValueType* _Buffer, SizeType _Count)
	{
		auto Ret = Empty<_MyValueType, 1, _MyDevice>(IDim(_Count));
		Ret.Fix(_Buffer, _Count);
		return Ret;
	}

	template <Device _MyDevice = Device::CPU, typename _MyArrayLike>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		std::enable_if_t<TypeTraits::IsArrayLikeValue<_MyArrayLike>&&
		std::is_copy_assignable_v<TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>>,
		Tensor<TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>,
		TypeTraits::ExtractRankValue<_MyArrayLike>,
		_MyDevice>> CopyFromArrayLike(const _MyArrayLike& _Array)
	{
		using ValueType = TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>;
		static_assert(std::is_copy_assignable_v<ValueType>, "The value type of the array-like object must be copy assignable.");
		constexpr size_t Rank = TypeTraits::ExtractRankValue<_MyArrayLike>;

		const auto& _Shape = GetAllShapesOfArrayLikeType<_MyArrayLike>;
		auto Ret = Empty<ValueType, Rank, _MyDevice>(_Shape);
		const auto TotalSize = _Shape.Multiply();
		if constexpr (TypeTraits::IsArrayValue<_MyArrayLike>)
			Ret.Fix(&_Array[0], TotalSize);
		else if (sizeof(_MyArrayLike) == sizeof(ValueType) * TotalSize)
			Ret.Fix((const ValueType*)&_Array, TotalSize);
		else
			_D_Dragonian_Lib_Throw_Exception("The array-like object is not compatible with the tensor.");
		return Ret;
	}

	template <Device _MyDevice = Device::CPU, typename _MyArrayLike>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		std::enable_if_t<TypeTraits::IsArrayLikeValue<_MyArrayLike>&&
		std::is_move_assignable_v<TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>>,
		Tensor<TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>,
		TypeTraits::ExtractRankValue<_MyArrayLike>,
		_MyDevice>> MoveFromArrayLike(const _MyArrayLike& _Array)
	{
		using ValueType = TypeTraits::ExtractInnerTypeOfArrayType<_MyArrayLike>;
		static_assert(std::is_move_assignable_v<ValueType>, "The value type of the array-like object must be move constructible.");
		constexpr size_t Rank = TypeTraits::ExtractRankValue<_MyArrayLike>;

		const auto& _Shape = GetAllShapesOfArrayLikeType<_MyArrayLike>;
		auto Ret = Empty<ValueType, Rank, _MyDevice>(_Shape);
		const auto TotalSize = _Shape.Multiply();
		if constexpr (TypeTraits::IsArrayValue<_MyArrayLike>)
			Ret.MoveFix(&_Array[0], TotalSize);
		else if (sizeof(_MyArrayLike) == sizeof(ValueType) * TotalSize)
			Ret.MoveFix((const ValueType*)&_Array, TotalSize);
		else
			_D_Dragonian_Lib_Throw_Exception("The array-like object is not compatible with the tensor.");
		return Ret;
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice,
		typename = std::enable_if_t<std::is_copy_assignable_v<_MyValueType>>>
	decltype(auto) Copy(const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
	{
		return _Tensor.Clone();
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice,
		typename = std::enable_if_t<std::is_copy_assignable_v<_MyValueType>>>
	decltype(auto) Clone(const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
	{
		return _Tensor.Clone();
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice,
		typename = std::enable_if_t<std::is_copy_assignable_v<_MyValueType>>>
	decltype(auto) MakeContinuous(Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
	{
		return _Tensor.MakeContinuous();
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice,
		typename = std::enable_if_t<std::is_copy_assignable_v<_MyValueType>>>
	decltype(auto) Continuous(const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
	{
		return _Tensor.Continuous();
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice, size_t _TRank>
	decltype(auto) View(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		const Dimensions<_TRank>& _ViewShape
	)
	{
		return _Tensor.View(_ViewShape);
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice, typename... _Args>
	decltype(auto) View(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		_Args... _ViewShape
	)
	{
		if constexpr (sizeof...(_Args) == 0)
			return _Tensor.View();
		else
			return _Tensor.View( _ViewShape... );
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice>
	decltype(auto) UnSqueeze(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		size_t _Axis
	)
	{
		return _Tensor.UnSqueeze(_Axis);
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice>
	decltype(auto) Squeeze(const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor, size_t _Axis)
	{
		return _Tensor.Squeeze(_Axis);
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice,
		typename = std::enable_if_t<(_NRank > 1)>>
	decltype(auto) Transpose(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		SizeType _Axis1 = -1, SizeType _Axis2 = -2
	)
	{
		return _Tensor.Transpose(_Axis1, _Axis2);
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice>
	decltype(auto) Permute(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		const Dimensions<_NRank>& _PremuteOrder
	)
	{
		return _Tensor.Permute(_PremuteOrder);
	}

	template <typename _MyValueType, size_t _NRank, Device _MyDevice, typename... _Args,
		typename = std::enable_if_t<sizeof...(_Args) == _NRank>>
	decltype(auto) Permute(
		const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor,
		_Args... _PremuteOrder
	)
	{
		return _Tensor.Permute( _PremuteOrder... );
	}
}

template <typename _MyValueType, size_t _NRank, Device _MyDevice>
std::ostream& operator<<(std::ostream& _OStream, const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
{
	_OStream << _Tensor.CastToString();
	return _OStream;
}

template <typename _MyValueType, size_t _NRank, Device _MyDevice>
std::wostream& operator<<(std::wostream& _OStream, const Tensor<_MyValueType, _NRank, _MyDevice>& _Tensor)
{
	_OStream << _Tensor.CastToWideString();
	return _OStream;
}



_D_Dragonian_Lib_Space_End