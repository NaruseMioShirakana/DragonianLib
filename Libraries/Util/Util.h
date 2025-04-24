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

#include <stdexcept>
#include <string>
#include <filesystem>

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

// Define exception throwing macro with trace
#define _D_Dragonian_Error_Message_With_Trace(message) \
	_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Throw_Function_Impl(message, __FILE__, _D_Dragonian_Lib_Function_Signature, __LINE__)

// Define exception throwing macro with raw message and exception type
#define _D_Dragonian_Lib_Throw_Raw_Exception_Impl(message, exception_type) throw exception_type(message)

// Define exception throwing macro with raw message
#define _D_Dragonian_Lib_Throw_Raw_Exception(message) _D_Dragonian_Lib_Throw_Raw_Exception_Impl(message, std::exception)

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

struct NoneType {}; ///< None type
static constexpr NoneType None; ///< None constant
template <typename _Type>
constexpr bool operator==(const _Type&, const NoneType&)
{
	if constexpr (std::is_same_v<_Type, NoneType>)
		return true;
	else
		return false;
}

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
	if (Message.starts_with("[@"))
	{
		if (Message.starts_with(Prefix))
			return Message;
		return Prefix.substr(0, Prefix.length() - 2) + "\n " + Message.substr(1);
	}
	return Prefix + ' ' + Message;
}

_D_Dragonian_Lib_Space_End