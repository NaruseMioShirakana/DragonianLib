#pragma once
#include "Util.h"
#include "Libraries/Util/TypeTraits.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

template <class _ValueType, size_t _Rank>
class Array
{
public:
	static constexpr size_t _MyRank = _Rank;
	static_assert(_MyRank > 0, "The rank of the array must be greater than 0.");
	using ArrayType = _ValueType[_Rank];
	_D_Dragonian_Lib_Constexpr_Force_Inline Array& Assign(const _ValueType* _Right)
	{
		for (size_t i = 0; i < _Rank; ++i)
			_MyData[i] = *_Right++;
		return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static Array ConstantOf(_ValueType _Value)
	{
		Array _Tmp;
		for (size_t i = 0; i < _Rank; ++i)
			_Tmp._MyData[i] = _Value;
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void AssignConstant(_ValueType _Value)
	{
		for (size_t i = 0; i < _Rank; ++i)
			_MyData[i] = _Value;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::IsArithmeticValue<_Type>&&
		TypeTraits::IsSameTypeValue<_Type, _ValueType>,
		_ValueType> Sum() const
	{
		_ValueType _Sum = 0;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum += _MyData[i];
		return _Sum;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::IsArithmeticValue<_Type>&&
		TypeTraits::IsSameTypeValue<_Type, _ValueType>,
		_ValueType> InnerProduct(const Array<_Type, _Rank>& _Right) const
	{
		_ValueType _Sum = 0;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum += _MyData[i] * _Right._MyData[i];
		return _Sum;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
		TypeTraits::IsArithmeticValue<_Type>&&
		TypeTraits::IsSameTypeValue<_Type, _ValueType>,
		_ValueType> Multiply() const
	{
		_ValueType _Sum = 1;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum *= _MyData[i];
		return _Sum;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Size()
	{
		return _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* Data()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* Data() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* Begin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* End() const
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* Begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* End()
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& operator[](size_t _Index)
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& operator[](size_t _Index) const
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& At(size_t _Index)
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& At(size_t _Index) const
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& Front()
	{
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& Front() const
	{
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& Back()
	{
		return _MyData[_Rank - 1];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& Back() const
	{
		return _MyData[_Rank - 1];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* end()
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* begin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* end() const
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedBegin()
	{
		return _MyData + _Rank - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedEnd()
	{
		return &_MyData[0] - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedBegin() const
	{
		return _MyData + _Rank - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedEnd() const
	{
		return &_MyData[0] - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static bool Empty()
	{
		return _Rank == 0;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Rank()
	{
		return _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array<_ValueType, _Rank + 1> Insert(
		const _ValueType& _Value, size_t _Index
	) const
	{
		Array<_ValueType, _Rank + 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		_Tmp._MyData[_Index] = _Value;
		for (size_t i = _Index; i < _Rank; ++i)
			_Tmp._MyData[i + 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array<_ValueType, _Rank - 1> Erase(size_t _Index) const
	{
		Array<_ValueType, _Rank - 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		for (size_t i = _Index + 1; i < _Rank; ++i)
			_Tmp._MyData[i - 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline std::string ToString() const
	{
		std::string _Str = "[";
		for (size_t i = 0; i < _Rank; ++i)
		{
			_Str += std::to_string(_MyData[i]);
			if (i != _Rank - 1)
				_Str += ", ";
		}
		_Str += "]";
		return _Str;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline std::wstring ToWString() const
	{
		std::wstring _Str = L"[";
		for (size_t i = 0; i < _Rank; ++i)
		{
			_Str += std::to_wstring(_MyData[i]);
			if (i != _Rank - 1)
				_Str += L", ";
		}
		_Str += L"]";
		return _Str;
	}

	_ValueType _MyData[_Rank]; ///< Data of the dimensions
};

template<typename _Tp, typename... _Up>
Array(_Tp, _Up...)
-> ::DragonianLib::TemplateLibrary::Array<std::enable_if_t<(TypeTraits::IsSameTypeValue<_Tp, _Up> && ...), _Tp>,
         1 + sizeof...(_Up)>;

template <typename _ValueType, size_t _Size>
using MArray = Array<_ValueType, _Size>;

template <typename _Type, size_t _Rank>
struct _Impl_Static_Array_Type
{
	_Impl_Static_Array_Type() = delete;
	template<size_t _TRank, typename = std::enable_if_t<(_Rank > 1) && _TRank == _Rank - 1>>
		_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
			const _Type& _Value,
			const _Impl_Static_Array_Type<_Type, _TRank>& _Array
		)
	{
		for (size_t i = 0; i < _Array.Rank; ++i)
			Data[i + 1] = _Array.Data[i];
		Data[0] = _Value;
	}
	template<size_t _TRank, typename = std::enable_if_t<(_Rank > 1) && _TRank == _Rank - 1>>
		_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
			const _Impl_Static_Array_Type<_Type, _TRank>& _Array,
			const _Type& _Value
		)
	{
		for (size_t i = 0; i < _Array.Rank; ++i)
			Data[i] = _Array.Data[i];
		Data[_Array.Rank] = _Value;
	}

	template<typename = std::enable_if_t<_Rank == 1>>
	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Type& _Value,
		const _Impl_Static_Array_Type<_Type, 0>& _Array
	)
	{
		UNUSED(_Array);
		Data[0] = _Value;
	}
	template<typename = std::enable_if_t<_Rank == 1>>
	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Impl_Static_Array_Type<_Type, 0>& _Array,
		const _Type& _Value
	)
	{
		UNUSED(_Array);
		Data[0] = _Value;
	}

	template<typename = std::enable_if_t<_Rank == 1>>
	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Type& _Value
	)
	{
		Data[0] = _Value;
	}

	static constexpr size_t Rank = _Rank;
	Array<_Type, _Rank> Data;
};
template <typename _Type>
struct _Impl_Static_Array_Type<_Type, 0> {};

template <typename _Type>
struct ExtractAllShapesOfArrayLikeType;
template <typename _Type, size_t _Size>
struct ExtractAllShapesOfArrayLikeType<_Type[_Size]>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_Type> +1;
	template <typename = std::enable_if_t<(Rank > 0)>>
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape()
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_Size) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_Size),
				ExtractAllShapesOfArrayLikeType<_Type>::GetShape()
			);
			return Shape;
		}
	}
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractAllShapesOfArrayLikeType<_ObjType<_ValueType, _ValueSize>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
	template <typename = std::enable_if_t<(Rank > 0)>>
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape()
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_ValueSize) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_ValueSize),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractAllShapesOfArrayLikeType<_ObjType<_ValueSize, _ValueType>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
	template <typename = std::enable_if_t<(Rank > 0)>>
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape()
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_ValueSize) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_ValueSize),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};
template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArrayLikeValue<_Type>>>
const auto& GetAllShapesOfArrayLikeType = _D_Dragonian_Lib_TL_Namespace ExtractAllShapesOfArrayLikeType<_Type>::GetShape().Data;

template <typename _ValueType>
struct ExtractAllShapesOfInitializerList;
template <typename _ValueType>
struct ExtractAllShapesOfInitializerList<std::initializer_list<_ValueType>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
	template <typename = std::enable_if_t<(Rank > 0)>>
		static constexpr _Impl_Static_Array_Type<int64_t, Rank> GetShape(const std::initializer_list<_ValueType>& _Val)
	{
		if constexpr (Rank == 1)
		{
			_Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_Val.size()) };
			return Shape;
		}
		else
		{
			_Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_Val.size()),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};

_D_Dragonian_Lib_Template_Library_Space_End