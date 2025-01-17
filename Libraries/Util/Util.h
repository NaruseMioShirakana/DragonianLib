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
	if (Message.substr(0, 2) == "[@")
	{
		if (Message.substr(0, Prefix.length()) == Prefix)
			return Message;
		return Prefix.substr(0, Prefix.length() - 2) + "\n " + Message.substr(1);
	}
	return Prefix + ' ' + Message;
}

_D_Dragonian_Lib_Space_End