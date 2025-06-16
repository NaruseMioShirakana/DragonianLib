/**
 * @file Tensor.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Tensor of DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once

#include <deque>
#include <ranges>

#include "Libraries/Util/StringPreprocess.h"

#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"

_D_Dragonian_Lib_Space_Begin

static inline constexpr SizeType RangeBeginPos = INT64_MAX; ///< Begin index
static inline constexpr SizeType RangeEndPos = INT64_MIN; ///< End index
static inline SizeType ZeroConstantVal = 0; ///< None index

/**
 * @brief Enum class representing padding types.
 */
enum class PaddingType : uint8_t
{
	Zero, ///< Zero padding
	Constant, ///< Constant padding
	Reflect, ///< Reflect padding
	Cicular, ///< Circular padding
	Replicate ///< Replicate padding
};

/**
 * @brief Struct representing a range with begin, step, and end values.
 */
struct Range
{
	SizeType Begin = 0; ///< Begin value
	SizeType Step = 1; ///< Step value
	SizeType End = RangeEndPos; ///< End value

	_D_Dragonian_Lib_Constexpr_Force_Inline Range() = default;

	/**
	 * @brief Constructor for a none range.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(NoneType) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(nullptr_t) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(std::nullopt_t) {}

	_D_Dragonian_Lib_Constexpr_Force_Inline Range(SizeType _Val) : Begin(_Val), Step(_Val), End(_Val) {}

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const char* _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const wchar_t* _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const std::string& _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const std::wstring& _RangeArgs);

	template <size_t N>
	Range(const char(&_RangeArgs)[N]) : Range(&_RangeArgs[0]) {}

	template <size_t N>
	Range(const wchar_t(&_RangeArgs)[N]) : Range(&_RangeArgs[0]) {}

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End) {}

	/**
	 * @brief Constructor for a range with begin and end values.
	 * @param _Begin The begining value.
	 * @param _End The end value.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(SizeType _Begin, SizeType _End) :Begin(_Begin), End(_End) {}

	/**
	 * @brief Constructor for a range with none and end values.
	 * @param _End The end value.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(NoneType, SizeType _End) :End(_End) {}

	/**
	 * @brief Constructor for a range with begin and none values.
	 * @param _Begin The begining value.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline Range(SizeType _Begin, NoneType) :Begin(_Begin) {}

	/**
	 * @brief Reverse the range.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline void Reverse() { std::swap(Begin, End); Step = -Step; }

	std::string ToString() const
	{
		return "[" + std::to_string(Begin) + ":" + std::to_string(Step) + ":" +
			(End == RangeEndPos ? std::string("EndPos") : std::to_string(End)) +
			"]";
	}

	std::wstring ToWString() const
	{
		return L"[" + std::to_wstring(Begin) + L":" + std::to_wstring(Step) + L":" +
			(End == RangeEndPos ? std::wstring(L"EndPos") : std::to_wstring(End)) +
			L"]";
	}

	Range operator-() const { return { End, -Step, Begin }; }

	static Range Idx(SizeType Idx) { return { Idx, Idx, Idx }; }

private:
	void Parse(const std::string_view& _RangeArgs);
	void Parse(const std::wstring_view& _RangeArgs);
};

constexpr Range NPAD{ 0, 0, 0 }; ///< Zero padding count

template <size_t _NRank>
class SliceOptions : public IDLArray<Range, _NRank>
{

};
template <size_t _NRank>
class VRanges : public IDLArray<Range, _NRank>
{
	
};
template <size_t _NRank>
class PaddingCounts : public IDLArray<Range, _NRank>
{

};
template <size_t _NRank>
class Dimensions : public IDLArray<SizeType, _NRank>
{
public:
	using IDLArray<SizeType, _NRank>::_MyData;
	_D_Dragonian_Lib_Constexpr_Force_Inline Dimensions<_NRank + 1> Insert(
		const SizeType& _Value, size_t _Index
	) const
	{
		Dimensions<_NRank + 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		_Tmp._MyData[_Index] = _Value;
		for (size_t i = _Index; i < _NRank; ++i)
			_Tmp._MyData[i + 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Dimensions<_NRank - 1> Erase(size_t _Index) const
	{
		Dimensions<_NRank - 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		for (size_t i = _Index + 1; i < _NRank; ++i)
			_Tmp._MyData[i - 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static Dimensions ConstantOf(const SizeType& _Value)
	{
		Dimensions _Tmp;
		for (size_t i = 0; i < _NRank; ++i)
			_Tmp._MyData[i] = _Value;
		return _Tmp;
	}
};

using PadCount = Range;
template<typename... _Up>
SliceOptions(_Up&&...) -> ::DragonianLib::SliceOptions<sizeof...(_Up)>;
template<typename... _Up>
VRanges(_Up&&...) -> ::DragonianLib::VRanges<sizeof...(_Up)>;
template<typename... _Up>
PaddingCounts(_Up&&...) -> ::DragonianLib::PaddingCounts<sizeof...(_Up)>;
template<typename... _Up>
Dimensions(_Up&&...) -> ::DragonianLib::Dimensions<sizeof...(_Up)>;

namespace TypeTraits
{
	template <typename>
	struct IsRange : std::false_type {};
	template <>
	struct IsRange<Range> : std::true_type {};
	template <typename _Type>
	constexpr bool IsRangeValue = IsRange<std::remove_cv_t<_Type>>::value;
}

/**
 * @brief Set the random seed.
 * @param _Seed The seed value.
 */
void SetRandomSeed(SizeType _Seed);

/**
 * @brief Set the number of worker threads.
 * @param _ThreadCount The number of worker threads.
 */
void SetWorkerCount(SizeType _ThreadCount);

/**
 * @brief Set the task pool size.
 * @param _Size The size of the task pool.
 */
void SetTaskPoolSize(SizeType _Size);

/**
 * @brief Set the maximum task count per operator.
 * @param _MaxTaskCount The maximum task count per operator.
 */
void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount);

/**
 * @brief Enable the time logger in thread pool.
 * @param _Enable True to enable, false to disable.
 */
void EnableTimeLogger(bool _Enable);

/**
 * @brief Enable the instant run in thread pool.
 * @param _Enable True to enable, false to disable.
 */
void EnableInstantRun(bool _Enable);

template <size_t _NRank>
std::string ToString(const Dimensions<_NRank>& _Dimensions)
{
	if constexpr (_Dimensions.Empty())
		return "()";
	std::string Ret = "(";
	for (const auto& Dim : _Dimensions)
		Ret += std::to_string(Dim) + ", ";
	Ret.pop_back(); Ret.pop_back();
	Ret += ")";
	return Ret;
}

template <typename _Type>
_D_Dragonian_Lib_Constexpr_Force_Inline
decltype(auto) MaxOf(const _Type& _Left, const _Type& _Right) { return _Left > _Right ? _Left : _Right; }
template <typename _Type>
_D_Dragonian_Lib_Constexpr_Force_Inline
decltype(auto) MinOf(const _Type& _Left, const _Type& _Right) { return _Left < _Right ? _Left : _Right; }

_D_Dragonian_Lib_Space_End

template <size_t _Size>
struct DragonianLib::TypeTraits::ImplArrayTraits<DragonianLib::VRanges<_Size>>
{
	using Type = Range;
	static constexpr size_t Size = _Size;
};
template <size_t _Size>
constexpr bool DragonianLib::TypeTraits::IsArrayLike<DragonianLib::VRanges<_Size>> = true;
template <size_t _Size>
struct DragonianLib::TypeTraits::ImplArrayTraits<DragonianLib::PaddingCounts<_Size>>
{
	using Type = Range;
	static constexpr size_t Size = _Size;
};
template <size_t _Size>
constexpr bool DragonianLib::TypeTraits::IsArrayLike<DragonianLib::PaddingCounts<_Size>> = true;
template <size_t _Size>
struct DragonianLib::TypeTraits::ImplArrayTraits<DragonianLib::SliceOptions<_Size>>
{
	using Type = Range;
	static constexpr size_t Size = _Size;
};
template <size_t _Size>
constexpr bool DragonianLib::TypeTraits::IsArrayLike<DragonianLib::SliceOptions<_Size>> = true;
template <size_t _Size>
struct DragonianLib::TypeTraits::ImplArrayTraits<DragonianLib::Dimensions<_Size>>
{
	using Type = SizeType;
	static constexpr size_t Size = _Size;
};
template <size_t _Size>
constexpr bool DragonianLib::TypeTraits::IsArrayLike<DragonianLib::Dimensions<_Size>> = true;