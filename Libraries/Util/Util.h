/**
 * @file Util.h
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
 * @brief Utility functions and macros
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <stacktrace>
#include <string>
#include <filesystem>

// Define UNUSED macro for unused variables
#ifndef UNUSED
#define UNUSED(...) (void)(__VA_ARGS__)
#endif

#define _D_Dragonian_Lib DragonianLib

#define _D_Dragonian_Lib_Namespace _D_Dragonian_Lib::

// Define namespace macros
#define _D_Dragonian_Lib_Space_Begin namespace _D_Dragonian_Lib {

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
#define _D_Dragonian_Lib_Function_Signature __FUNCTION__
#elif defined(__GNUC__) || defined(__clang__)
#define _D_Dragonian_Lib_Function_Signature __PRETTY_FUNCTION__
#endif

#define _D_Dragonian_Constexpr_String(Str) \
	decltype(_D_Dragonian_Lib_Namespace ConstexprString::Create<(Str)>())

// Define exception throwing macro with trace
#ifdef DRAGONIANLIB_DEBUG_MACRO
#define _D_Dragonian_Error_Message_With_Trace(message) \
	_D_Dragonian_Lib_Namespace GetTraceBack<_D_Dragonian_Constexpr_String(__FILE__), (__LINE__)>().GetTraceBackString((message))
#else
#define _D_Dragonian_Error_Message_With_Trace(message) \
	_D_Dragonian_Lib_Namespace TraceBack(__FILE__, __LINE__, _D_Dragonian_Lib_Function_Signature).GetTraceBackString(message)
#endif

// Define general exception throwing macro
#ifdef DRAGONIANLIB_DEBUG_MACRO
#define _D_Dragonian_Lib_Throw_Exception(message) \
	throw _D_Dragonian_Lib_Namespace ErrorCode<_D_Dragonian_Constexpr_String(__FILE__), (__LINE__)>((message))
#else
#define _D_Dragonian_Lib_Throw_Exception(message) \
	throw _D_Dragonian_Lib_Namespace ErrorCode( \
		(message), \
		__FILE__, \
		__LINE__, \
		_D_Dragonian_Lib_Function_Signature \
	)
#endif

// Define not implemented error macro
#define _D_Dragonian_Lib_Not_Implemented_Error \
	_D_Dragonian_Lib_Throw_Exception("Not Implemented Error!")

// Define fatal error macro
#define _D_Dragonian_Lib_Fatal_Error \
	_D_Dragonian_Lib_Throw_Exception("Fatal Error!")

// Define assert macro
#define _D_Dragonian_Lib_Assert(Expr, Message) \
	if (!(Expr)) _D_Dragonian_Lib_Throw_Exception(Message)

// Define cuda error
#define _D_Dragonian_Lib_CUDA_Error \
	_D_Dragonian_Lib_Throw_Exception(cudaGetErrorString(cudaGetLastError()))

#define _D_Dragonian_Lib_Rethrow_Block(Expr) \
do{ \
	try \
	{ \
		Expr \
	} \
	catch(std::exception& _M_EXCEPT) \
	{ \
		_D_Dragonian_Lib_Throw_Exception(_M_EXCEPT.what()); \
	} \
} \
while (0) \

#define _D_Dragonian_Lib_Return_Exception_Block(Expr) \
do{ \
	try \
	{ \
		Expr \
	} \
	catch(std::exception& _M_EXCEPT) \
	{ \
		return _D_Dragonian_Error_Message_With_Trace(_M_EXCEPT.what()); \
	} \
} \
while (0) \

// Define registration layer macro
#define DragonianLibRegLayer(ModuleName, MemberName, ...) ModuleName MemberName{this, #MemberName, __VA_ARGS__}

#ifndef DRAGONIANLIB_ALLOC_ALIG
#define DRAGONIANLIB_ALLOC_ALIG 32
#endif
#ifndef DRAGONIANLIB_ALIG_DIM_SHAPE
#define DRAGONIANLIB_ALIG_DIM_SHAPE 8
#endif
#ifndef DRAGONIANLIB_CONT_THRESHOLD_FRONT
#define DRAGONIANLIB_CONT_THRESHOLD_FRONT 8
#endif
#ifndef DRAGONIANLIB_CONT_THRESHOLD_BACK
#define DRAGONIANLIB_CONT_THRESHOLD_BACK 32
#endif
#ifndef DRAGONIANLIB_EMPTY_CAPACITY
#define DRAGONIANLIB_EMPTY_CAPACITY 16
#endif
#ifndef DRAGONIANLIB_PADDING_COUNT
#define DRAGONIANLIB_PADDING_COUNT 64000
#endif
#ifndef DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE
#define DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE 8192
#endif
#ifndef DRAGONIANLIB_NAME_MAX_SIZE
#define DRAGONIANLIB_NAME_MAX_SIZE 1024
#endif

_D_Dragonian_Lib_Space_Begin

struct NoneType
{
	constexpr NoneType() {}
	constexpr NoneType(nullptr_t) {}
	constexpr NoneType(std::nullopt_t) {}
}; ///< None type
static constexpr NoneType None; ///< None constant
template <typename _Type>
constexpr bool operator==(const _Type&, const NoneType&)
{
	if constexpr (std::is_same_v<_Type, NoneType>)
		return true;
	else
		return false;
}

namespace ConstexprString
{
	template <size_t N>
	struct String
	{
		constexpr static size_t _MySize = N;
		constexpr String(const char(&Arr)[N])
		{
			for (size_t i = 0; i < N; ++i)
				_MyString[i] = Arr[i];
		}
		char _MyString[N];
	};

	template <char... Chars>
	struct Struct
	{
		static constexpr char Value[]{
			Chars...
		};
	};

	template <size_t Index, size_t N>
	constexpr char Get(const char(&Arr)[N])
	{
		return Arr[Index];
	}

	template <String Str, size_t... I>
	auto Create(std::index_sequence<I...>)
	{
		return Struct<Get<I>(Str._MyString)...>{};
	}

	template <String Str>
	auto Create()
	{
		return Create<Str>(
			std::make_index_sequence<Str._MySize>{}
		);
	}
}

class TraceBack
{
public:
#ifdef DRAGONIANLIB_DEBUG_MACRO
	TraceBack(ptrdiff_t BeginPos = 1)
	{
		const auto Trace = std::stacktrace::current();
		_MyTraceBack += "$Error Occurred:\n";
		for (auto Iter = Trace.begin() + BeginPos; Iter < Trace.end(); ++Iter)
		{
			auto SourceFile = Iter->source_file();
			std::ranges::replace(SourceFile, '\\', '/');
			if (SourceFile.starts_with(__DRAGONIANLIB_SOURCE_DIRECTORY))
				SourceFile = SourceFile.substr(sizeof(__DRAGONIANLIB_SOURCE_DIRECTORY));
			if (SourceFile.empty())
				SourceFile = "Unknown";
			_MyTraceBack += " @[SourceFile: \"" + SourceFile + "\"; " +
				"Line: <" + std::to_string(Iter->source_line()) + ">; " +
				"Function: \"" + Iter->description() + "\"]:\n";
		}
		_MyTraceBack += " Status Message: ";
	}
#endif

	TraceBack(const char* _File, int _Line, const char* _Function)
	{
		_MyTraceBack += "$Error Occurred:\n";
		std::string SourceFile = _File;
		std::ranges::replace(SourceFile, '\\', '/');
		if (SourceFile.starts_with(__DRAGONIANLIB_SOURCE_DIRECTORY))
			SourceFile = SourceFile.substr(sizeof(__DRAGONIANLIB_SOURCE_DIRECTORY));
		if (SourceFile.empty())
			SourceFile = "Unknown";
		_MyTraceBack += " @[SourceFile: \"" + SourceFile + "\"; " +
			"Line: <" + std::to_string(_Line) + ">; " +
			"Function: \"" + _Function + "\"]:\n";
	}

	std::string GetTraceBackString(
#ifdef DRAGONIANLIB_DEBUG_MACRO
		std::string Message
#else
		const std::string& Message
#endif
	) const
	{
		if (!Message.starts_with("$Error Occurred:\n"))
			return _MyTraceBack + Message;
#ifdef DRAGONIANLIB_DEBUG_MACRO
		return Message;
#else
		return _MyTraceBack + Message.substr(sizeof("$Error Occurred:"));
#endif
	}
private:
	std::string _MyTraceBack;
};

#ifdef DRAGONIANLIB_DEBUG_MACRO
template <typename, int>
decltype(auto) GetTraceBack(ptrdiff_t BeginPos = 2)
{
	static TraceBack TraceBack(BeginPos);
	return TraceBack;
}
#endif

#ifdef DRAGONIANLIB_DEBUG_MACRO
template <typename _MyFile, int _MyLine>
#endif
class ErrorCode : public std::exception
{
public:
	ErrorCode() = default;

	ErrorCode(
		const char* _Message
#ifndef DRAGONIANLIB_DEBUG_MACRO
		, const char* _FileName
		, int _Line
		, const char* _Function
#endif
	) : std::exception("Dragonian Lib Exception")
	{
		_MyMessage = _Message;
#ifdef DRAGONIANLIB_DEBUG_MACRO
		if (!_MyMessage.starts_with("$Error Occurred:\n"))
			_MyMessage = GetTraceBack<_MyFile, _MyLine>(3).GetTraceBackString(_MyMessage);
#else
		_MyMessage = TraceBack(_FileName, _Line, _Function).GetTraceBackString(_MyMessage);
#endif
	}
	ErrorCode(
		std::string _Message
#ifndef DRAGONIANLIB_DEBUG_MACRO
		, const char* _FileName
		, int _Line
		, const char* _Function
#endif
	) : std::exception("Dragonian Lib Exception")
	{
		_MyMessage = std::move(_Message);
#ifdef DRAGONIANLIB_DEBUG_MACRO
		if (!_MyMessage.starts_with("$Error Occurred:\n"))
			_MyMessage = GetTraceBack<_MyFile, _MyLine>(3).GetTraceBackString(_MyMessage);
#else
		_MyMessage = TraceBack(_FileName, _Line, _Function).GetTraceBackString(_MyMessage);
#endif
	}
	~ErrorCode() noexcept override = default;
	const char* what() const noexcept override
	{
		return _MyMessage.c_str();
	}
	ErrorCode(const ErrorCode&) = default;
	ErrorCode(ErrorCode&&) = default;
	ErrorCode& operator=(const ErrorCode&) = default;
	ErrorCode& operator=(ErrorCode&&) = default;
private:
	std::string _MyMessage;
};

enum class ComInitializeFlag : uint8_t
{
	COINIT_APARTMENTTHREADED,
	COINIT_MULTITHREADED,
	COINIT_DISABLE_OLE1DDE,
	COINIT_SPEED_OVER_MEMORY,
};

template <bool _Fs, bool... _Rest>
struct BoolConditionAnd
{
	static constexpr bool Cond = _Fs && BoolConditionAnd<_Rest...>::Cond;
};
template <bool _Fs>
struct BoolConditionAnd<_Fs>
{
	static constexpr bool Cond = _Fs;
};
template <bool... _Conditions>
constexpr bool BoolConditionAndValue = BoolConditionAnd<_Conditions...>::Cond;
template <bool _Fs, bool... _Rest>
struct BoolConditionOr
{
	static constexpr bool Cond = _Fs || BoolConditionOr<_Rest...>::Cond;
};
template <bool _Fs>
struct BoolConditionOr<_Fs>
{
	static constexpr bool Cond = _Fs;
};
template <bool... _Conditions>
constexpr bool BoolConditionOrValue = BoolConditionOr<_Conditions...>::Cond;

template <typename... _Type>
void ExpandExpression(_Type&&...) {}

constexpr decltype(auto) ConstexprSum()
{
	return 0;
}
template <typename _Type>
constexpr decltype(auto) ConstexprSum(_Type&& _First)
{
	return std::forward<_Type>(_First);
}
template <typename _Type, typename... _Rest>
constexpr decltype(auto) ConstexprSum(_Type&& _Arg, _Rest&&... _Args)
{
	return std::forward<_Type>(_Arg) + ConstexprSum(std::forward<_Rest>(_Args)...);
}
template <typename... _Type>
constexpr decltype(auto) ExpandSum(_Type&&... _Args)
{
	return ConstexprSum(std::forward<_Type>(_Args)...);
}

constexpr decltype(auto) ConstexprMul()
{
	return 0;
}
template <typename _Type>
constexpr decltype(auto) ConstexprMul(_Type&& _First)
{
	return std::forward<_Type>(_First);
}
template <typename _Type, typename... _Rest>
constexpr decltype(auto) ConstexprMul(_Type&& _Arg, _Rest&&... _Args)
{
	return std::forward<_Type>(_Arg) * ConstexprMul(std::forward<_Rest>(_Args)...);
}
template <typename... _Type>
constexpr decltype(auto) ExpandMul(_Type&&... _Args)
{
	return ConstexprMul(std::forward<_Type>(_Args)...);
}

struct HResult
{
	int64_t Value;
	operator bool() const;
};

uint32_t Cvt2tagCOINIT(ComInitializeFlag _Flag);

HResult ComInitialize(uint32_t _Flag = Cvt2tagCOINIT(ComInitializeFlag::COINIT_MULTITHREADED));

void ComUninitialize();

_D_Dragonian_Lib_Space_End