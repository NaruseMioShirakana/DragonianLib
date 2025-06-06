﻿/**
 * @file StringPreprocess.h
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
 * @brief String utilities for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <regex>

#include "Libraries/Util/TypeTraits.h"

_D_Dragonian_Lib_Space_Begin

namespace PreDefinedRegex
{
	static inline const std::wstring Chinese =
		LR"(([\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u2F00-\u2FDF\u2E80-\u2EFF\u31C0-\u31EF\u2FF0-\u2FFF]))";
	static inline const std::wstring Japanese =
		LR"([\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF])";
	static inline const std::wstring FullWidth =
		LR"([\uFF00-\uFFEF])";
	static inline const std::wstring Alphabet =
		LR"([a-zA-Z])";
	static inline const std::wstring ChineseAndJapanese =
		LR"(([\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u2F00-\u2FDF\u2E80-\u2EFF\u31C0-\u31EF\u2FF0-\u2FFF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]))";
	static inline const std::wstring Number =
		LR"([0-9])";
	static inline const std::wstring Real =
		LR"([0-9]+(?:\.[0-9]+)?)";
	static inline const std::wstring Integer =
		LR"([0-9]+)";
	static inline const std::wstring AllSymbol =
		L"[ !@#$%^&*()_+\\-=`~,./;'\\[\\]<>?:\"{}|\\\\。？！，、；：“”‘’『』「」（）〔〕【】─…·—～《》〈〉]";
	static inline const std::wstring NewLine =
		L"[\\n\\r]";
	static inline const std::wstring Space =
		L"[\\s]";
	static inline const std::wstring ChineseNumber[]{
		L"零",L"一",L"二",L"三",L"四",L"五",L"六",L"七",L"八",L"九"
	};
	static inline const std::wstring ChineseNumberDigit[]{
		L"",L"十",L"百",L"千",L"万",L"十万",L"百万",L"千万",L"亿", L"十亿", L"百亿", L"千亿", L"兆", L"十兆", L"百兆", L"千兆"
	};

	static inline const std::wregex ChineseRegex = std::wregex(Chinese);
	static inline const std::wregex NumberRegex = std::wregex(Number);
	static inline const std::wregex AlphabetRegex = std::wregex(Alphabet);
	static inline const std::wregex JapaneseRegex = std::wregex(Japanese);
	static inline const std::wregex AllSymbolRegex = std::wregex(AllSymbol);
	static inline const std::wregex FullWidthRegex = std::wregex(FullWidth);
	static inline const std::wregex ChineseAndJapaneseRegex = std::wregex(ChineseAndJapanese);
	static inline const std::wregex ChineseStringRegex = std::wregex(Chinese + L"+");
	static inline const std::wregex NumberStringRegex = std::wregex(Number + L"+");
	static inline const std::wregex AlphabetStringRegex = std::wregex(Alphabet + L"+");
	static inline const std::wregex JapaneseStringRegex = std::wregex(Japanese + L"+");
	static inline const std::wregex AllSymbolGroupRegex = std::wregex(AllSymbol + L"+");
	static inline const std::wregex FullWidthStringRegex = std::wregex(FullWidth + L"+");
	static inline const std::wregex ChineseAndJapaneseStringRegex = std::wregex(ChineseAndJapanese + L"+");
	static inline const std::wregex IntegerRegex = std::wregex(Integer);
	static inline const std::wregex RealRegex = std::wregex(Real);
	static inline const std::wregex NewLineRegex = std::wregex(NewLine);
	static inline const std::wregex SpaceRegex = std::wregex(Space);
	static inline const std::wregex PunctuationRegex = std::wregex(AllSymbol);
	static inline const std::wregex PunctuationGroupRegex = std::wregex(AllSymbol + L"+");
}

std::string WideStringToUTF8(const std::string& input);
std::string WideStringToUTF8(const std::wstring& input);

std::string UnicodeToAnsi(const std::string& input);
std::string UnicodeToAnsi(const std::wstring& input);

std::wstring UTF8ToWideString(const std::wstring& input);
std::wstring UTF8ToWideString(const std::string& input);

template <typename _Type>
decltype(auto) CvtToString(const _Type& _Value)
{
	if constexpr (TypeTraits::IsComplexValue<_Type>)
		return std::to_string(_Value.real()) + " + " + std::to_string(_Value.imag()) + "i";
	else if constexpr (requires(const _Type & _Tmp) { { std::to_string(_Tmp) }; })
		return std::to_string(_Value);
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.to_string() } -> TypeTraits::IsType<std::string>; })
		return _Value.to_string();
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.to_string() } -> TypeTraits::IsType<const char*>; })
		return std::string(_Value.to_string());
	else if constexpr (requires(const _Type & _Tmp) { { std::string(_Tmp) }; })
		return std::string(_Value);
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.c_str() } -> TypeTraits::IsType<const char*>; })
		return std::string(_Value.c_str());
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.str() } -> TypeTraits::IsType<std::string>; })
		return _Value.str();
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.str() } -> TypeTraits::IsType<const char*>; })
		return std::string(_Value.str());
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.string() } -> TypeTraits::IsType<std::string>; })
		return _Value.string();
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.string() } -> TypeTraits::IsType<const char*>; })
		return std::string(_Value.string());
	else if constexpr (requires(const _Type & _Tmp) { { std::to_wstring(_Tmp) }; })
		return WideStringToUTF8(std::to_wstring(_Value));
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.to_string() } -> TypeTraits::IsType<std::wstring>; })
		return WideStringToUTF8(_Value.to_string());  // NOLINT(bugprone-branch-clone)
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.to_string() } -> TypeTraits::IsType<const wchar_t*>; })
		return WideStringToUTF8(_Value.to_string());
	else if constexpr (requires(const _Type & _Tmp) { { std::wstring(_Tmp) }; })
		return WideStringToUTF8(std::wstring(_Value));
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.c_str() } -> TypeTraits::IsType<const wchar_t*>; })
		return WideStringToUTF8(_Value.c_str());
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.str() } -> TypeTraits::IsType<std::wstring>; } || requires(const _Type & _Tmp) { { _Tmp.str() } -> TypeTraits::IsType<const wchar_t*>; })
		return WideStringToUTF8(_Value.str());
	else if constexpr (requires(const _Type & _Tmp) { { _Tmp.string() } -> TypeTraits::IsType<std::wstring>; } || requires(const _Type & _Tmp) { { _Tmp.string() } -> TypeTraits::IsType<const wchar_t*>; })
		return WideStringToUTF8(_Value.string());
	else
		return "UnknownObject";
}

template <typename _Type>
_Type ToLowerString(const _Type& _String) requires (TypeTraits::IsCppStringValue<_Type>)
{
	_Type LowerString = _String;
	std::transform(LowerString.begin(), LowerString.end(), LowerString.begin(), ::tolower);
	return LowerString;
}

template <typename _Type>
_Type ToUpperString(const _Type& _String) requires (TypeTraits::IsCppStringValue<_Type>)
{
	_Type UpperString = _String;
	std::transform(UpperString.begin(), UpperString.end(), UpperString.begin(), ::toupper);
	return UpperString;
}

std::wstring Number2Chinese(const std::wstring& _Number);

UInt32 U16Word2Unicode(const std::wstring& input);

_D_Dragonian_Lib_Space_End
