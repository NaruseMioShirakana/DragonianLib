/**
 * FileName: Base.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include "TypeTraits.h"

// Define UNUSED macro for unused variables
#ifndef UNUSED
#define UNUSED(...) (void)(__VA_ARGS__)
#endif

#define _D_Dragonian_Lib_Namespace ::DragonianLib::

// Define namespace macros
#define _D_Dragonian_Lib_Space_Begin namespace DragonianLib {

// Define namespace end macro
#define _D_Dragonian_Lib_Space_End }

// Define Nodiscard macro
#define _D_Dragonian_Lib_No_Discard [[nodiscard]]

// Define Force Inline macro
#ifdef _MSC_VER
#define _D_Dragonian_Lib_Force_Inline __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define _D_Dragonian_Lib_Force_Inline __attribute__((always_inline)) inline
#else
#define _D_Dragonian_Lib_Force_Inline inline
#endif

#define _D_Dragonian_Lib_Constexpr_Force_Inline constexpr _D_Dragonian_Lib_Force_Inline

// Define exception throwing macro
#ifdef _MSC_VER
#define _D_Dragonian_Lib_Function_Signature __FUNCSIG__
#define _D_Dragonian_Lib_Throw_Impl(message, exception_type) throw exception_type(_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Throw_Function_Impl(message, __FILE__, _D_Dragonian_Lib_Function_Signature, __LINE__).c_str())
#elif defined(__GNUC__) || defined(__clang__)
#define _D_Dragonian_Lib_Function_Signature __PRETTY_FUNCTION__
#define _D_Dragonian_Lib_Throw_Impl(message, exception_type) throw exception_type(_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Throw_Function_Impl(message, __FILE__, _D_Dragonian_Lib_Function_Signature, __LINE__).c_str())
#endif

// Define exception throwing macro(without function name)
#define _D_Dragonian_Lib_Throw_Impl_With_Inline_Function(message, exception_type) throw exception_type(_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Throw_Function_Impl(message, __FILE__, "Inlined", __LINE__).c_str())

// Define general exception throwing macro
#define _D_Dragonian_Lib_Throw_Exception(message) _D_Dragonian_Lib_Throw_Impl(message, std::exception)

// Define exception throwing macro(without function name)
#define _D_Dragonian_Lib_Throw_With_Inline_Function(message) _D_Dragonian_Lib_Throw_Impl_With_Inline_Function(message, std::exception)

// Define not implemented error macro
#define _D_Dragonian_Lib_Not_Implemented_Error _D_Dragonian_Lib_Throw_Exception("Not Implemented Error!")

// Define fatal error macro
#define _D_Dragonian_Lib_Fatal_Error _D_Dragonian_Lib_Throw_Exception("Fatal Error!")

// Define assert macro
#define _D_Dragonian_Lib_Assert(Expr, Message) if (!(Expr)) _D_Dragonian_Lib_Throw_Exception(Message)

// Define cuda error
#define _D_Dragonian_Lib_CUDA_Error _D_Dragonian_Lib_Throw_Exception(cudaGetErrorString(cudaGetLastError()))

// Define registration layer macro
#define DragonianLibRegLayer(ModuleName, MemberName, ...) ModuleName MemberName{this, #MemberName, __VA_ARGS__}

_D_Dragonian_Lib_Space_Begin

//***************************************************Types*********************************************************//

template <typename _ValueType, size_t _Rank>
struct IDLArray
{
	static constexpr size_t _MyRank = _Rank;
	static_assert(_Rank > 0, "The rank of the array must be greater than 0.");
	using ArrayType = _ValueType[_Rank];
	_D_Dragonian_Lib_Constexpr_Force_Inline IDLArray& Assign(const _ValueType* _Right)
	{
		for (size_t i = 0; i < _Rank; ++i)
			_MyData[i] = *_Right++;
		return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static IDLArray ConstantOf(_ValueType _Value)
	{
		IDLArray _Tmp;
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
		_ValueType> InnerProduct(const IDLArray<_Type, _Rank>& _Right) const
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
		return _MyData - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedBegin() const
	{
		return _MyData + _Rank - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedEnd() const
	{
		return _MyData - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static bool Empty()
	{
		return _Rank == 0;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Rank()
	{
		return _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline IDLArray<_ValueType, _Rank + 1> Insert(
		const _ValueType& _Value, size_t _Index
	) const
	{
		IDLArray<_ValueType, _Rank + 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		_Tmp._MyData[_Index] = _Value;
		for (size_t i = _Index; i < _Rank; ++i)
			_Tmp._MyData[i + 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline IDLArray<_ValueType, _Rank - 1> Erase(size_t _Index) const
	{
		IDLArray<_ValueType, _Rank - 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		for (size_t i = _Index + 1; i < _Rank; ++i)
			_Tmp._MyData[i - 1] = _MyData[i];
		return _Tmp;
	}
	_ValueType _MyData[_Rank]; ///< Data of the dimensions
};

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
	IDLArray<_Type, _Rank> Data;
};
template <typename _Type>
struct _Impl_Static_Array_Type<_Type, 0> {};

template <typename _Type>
struct ExtractAllShapesOfArrayLikeType;
template <typename _Type, size_t _Size>
struct ExtractAllShapesOfArrayLikeType<_Type[_Size]>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_Type> + 1;
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
const auto& GetAllShapesOfArrayLikeType = _D_Dragonian_Lib_Namespace ExtractAllShapesOfArrayLikeType<_Type>::GetShape().Data;

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

//***************************************************Base*********************************************************//

/**
 * @brief Get error message with file path, function name and line number
 * @param Message Error message
 * @param Path File path
 * @param Function Function name
 * @param Line Line number
 * @return Error message with file path, function name and line number
 */
_D_Dragonian_Lib_Force_Inline std::string _Impl_Dragonian_Lib_Throw_Function_Impl(const std::string& Message, const char* Path, const char* Function, int Line)
{
	const std::string Prefix =
		std::string("[@file: \"") + std::filesystem::path(Path).filename().string() + "\"; " +
		"function: \"" + Function + "\"; " +
		"line: " + std::to_string(Line) + "]:";
	if (Message.substr(0, 2) == "[@")
	{
		if (Message.substr(0, Prefix.length()) == Prefix)
			return Message;
		return Prefix.substr(0, Prefix.length() - 2) + "\n " + Message.substr(1);
	}
	return Prefix + ' ' + Message;
}

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
// Define WeightHeader struct
struct WeightHeader
{
	Int64 Shape[8] = { 0,0,0,0,0,0,0,0 };
	char LayerName[DRAGONIANLIB_NAME_MAX_SIZE];
	char Type[16];
};
#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

// Define WeightData struct
struct WeightData
{
	WeightHeader Header_;
	std::vector<Byte> Data_;
	std::vector<Int64> Shape_;
	std::string Type_, LayerName_;
};

// Type alias for dictionary
using DictType = std::unordered_map<std::string, WeightData>;

/**
 * @brief Get global enviroment folder
 * @return global enviroment folder
 */
std::wstring GetCurrentFolder();

/**
 * @brief Set global enviroment folder
 * @param _Folder Folder to set
 */
void SetGlobalEnvDir(const std::wstring& _Folder);

/**
 * @class FileGuard
 * @brief RAII File
 */
class FileGuard
{
public:
	FileGuard() = default;
	~FileGuard();
	FileGuard(const std::wstring& _Path, const std::wstring& _Mode);
	FileGuard(const std::wstring& _Path, const wchar_t* _Mode);
	FileGuard(const wchar_t* _Path, const wchar_t* _Mode);
	FileGuard(const FileGuard& _Left) = delete;
	FileGuard& operator=(const FileGuard& _Left) = delete;
	FileGuard(FileGuard&& _Right) noexcept;
	FileGuard& operator=(FileGuard&& _Right) noexcept;

	/**
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const std::wstring& _Path, const std::wstring& _Mode);

	/**
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const std::wstring& _Path, const wchar_t* _Mode);

	/**
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const wchar_t* _Path, const wchar_t* _Mode);

	/**
	 * @brief Close file
	 */
	void Close();

	/**
	 * @brief Get file pointer
	 * @return file pointer
	 */
	operator FILE* () const;

	/**
	 * @brief Check if file is enabled
	 * @return true if file is enabled
	 */
	_D_Dragonian_Lib_No_Discard bool Enabled() const;

	void Seek(long _Offset, int _Origin) const;
	size_t Tell() const;

	size_t Read(void* _Buffer, size_t _BufferSize, size_t _ElementSize, size_t _Count = 1) const;
	size_t Write(const void* _Buffer, size_t _ElementSize, size_t _Count = 1) const;

	FileGuard& operator<<(const std::string& _Str);
	FileGuard& operator<<(const std::wstring& _Str);
	FileGuard& operator<<(const char* _Str);
	FileGuard& operator<<(const wchar_t* _Str);
	FileGuard& operator<<(char _Ch);
	FileGuard& operator<<(wchar_t _Ch);

private:
	FILE* file_ = nullptr;
};

template <typename _Func>
struct TidyGuard
{
	TidyGuard() = delete;
	TidyGuard(_Func Fn) : Fn_(Fn) {}
	~TidyGuard() { Fn_(); }
private:
	_Func Fn_;
	TidyGuard(const TidyGuard&) = delete;
	TidyGuard& operator=(const TidyGuard&) = delete;
	TidyGuard(TidyGuard&&) = delete;
	TidyGuard& operator=(TidyGuard&&) = delete;
};

enum class FloatPrecision
{
	BFloat16,
	Float16,
	Float32
};

template <typename _Type>
decltype(auto) CvtToString(const _Type& _Value)
{
	if constexpr (TypeTraits::IsComplexValue<_Type>)
		return std::to_string(_Value.real()) + " + " + std::to_string(_Value.imag()) + "i";
	else if constexpr (TypeTraits::IsArithmeticValue<_Type>)
		return std::to_string(_Value);
	else if constexpr (TypeTraits::IsStringValue<_Type>)
		return _Value;
	else if constexpr (TypeTraits::CouldBeConvertedFromValue<std::string, _Type> ||
		TypeTraits::CouldBeConvertedFromValue<const char*, _Type>)
		return std::string(_Value);
	else if constexpr (TypeTraits::CouldBeConvertedFromValue<std::wstring, _Type> ||
		TypeTraits::CouldBeConvertedFromValue<const wchar_t*, _Type>)
		return std::wstring(_Value);
	else
		return _Value.to_string();
}

_D_Dragonian_Lib_Space_End
