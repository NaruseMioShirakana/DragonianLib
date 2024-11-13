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
#include <complex>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

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
#define _D_Dragonian_Lib_Throw_Impl(message, exception_type) throw exception_type(_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Throw_Function_Impl(message, __FILE__, __FUNCTION__, __LINE__).c_str())

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

//*****************************************************Types********************************************************//
/**
 * @struct float16_t
 * @brief Half precision floating point struct
 */
struct float16_t {
	float16_t(float _Val);
	float16_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val[2];
	static uint16_t float32_to_float16(uint32_t f32);
	static uint32_t float16_to_float32(uint16_t f16);
};

/**
 * @struct float8_t
 * @brief 8-bit floating point struct
 */
struct float8_t
{
	float8_t(float _Val);
	float8_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val;
};

/**
 * @struct bfloat16_t
 * @brief bfloat16 struct
 */
struct bfloat16_t
{
	bfloat16_t(float _Val);
	bfloat16_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val[2];
};

using Int8 = int8_t; ///< 8-bit integer
using Int16 = int16_t; ///< 16-bit integer
using Int32 = int32_t; ///< 32-bit integer
using Int64 = int64_t; ///< 64-bit integer
using Float8 = float8_t; ///< 8-bit floating point
using BFloat16 = bfloat16_t; ///< bfloat16
using Float16 = float16_t; ///< Half precision floating point
using Float32 = float; ///< 32-bit floating point
using Float64 = double; ///< 64-bit floating point
using Byte = unsigned char; ///< Byte
using LPVoid = void*; ///< Pointer to void
using CPVoid = const void*; ///< Constant pointer to void
using UInt8 = uint8_t; ///< 8-bit unsigned integer
using UInt16 = uint16_t; ///< 16-bit unsigned integer
using UInt32 = uint32_t; ///< 32-bit unsigned integer
using UInt64 = uint64_t; ///< 64-bit unsigned integer
using Complex32 = std::complex<float>; ///< 32-bit complex
using Complex64 = std::complex<double>; ///< 64-bit complex
struct NoneType {}; ///< None type
static constexpr NoneType None; ///< None constant

//**************************************************Va List********************************************************//
template<typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Va_List { constexpr static int64_t _Size = sizeof...(_ArgTypes); };

//***********************************************Constexpr Decltype************************************************//
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
struct _Impl_Dragonian_Lib_Constexpr_Decltype;
template <typename _Tyt, typename _Tyf>
struct _Impl_Dragonian_Lib_Constexpr_Decltype<true, _Tyt, _Tyf> { using _Decltype = _Tyt; };
template <typename _Tyt, typename _Tyf>
struct _Impl_Dragonian_Lib_Constexpr_Decltype<false, _Tyt, _Tyf> { using _Decltype = _Tyf; };
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
using _Impl_Dragonian_Lib_Constexpr_Decltype_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Constexpr_Decltype<_Test, _Tyt, _Tyf>::_Decltype;
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
using _Impl_Dragonian_Lib_Conditional_t = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Constexpr_Decltype_t<_Test, _Tyt, _Tyf>;

template <typename _Ty1, typename _Ty2>
struct _Impl_Dragonian_Lib_Constexpr_Is_Same_Type { constexpr static bool _IsSame = false; };
template <typename _Ty1>
struct _Impl_Dragonian_Lib_Constexpr_Is_Same_Type<_Ty1, _Ty1> { constexpr static bool _IsSame = true; };
template <typename _Ty1, typename _Ty2>
constexpr bool _Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Constexpr_Is_Same_Type<_Ty1, _Ty2>::_IsSame;

template <typename _Ty1, typename ..._Types>
struct _Impl_Dragonian_Lib_Is_Any_Of { constexpr static bool _IsAnyOf = false; };
template <typename _Ty1, typename _Ty2, typename ..._Types>
struct _Impl_Dragonian_Lib_Is_Any_Of<_Ty1, _Ty2, _Types...> { constexpr static bool _IsAnyOf = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Ty1, _Ty2> || _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of<_Ty1, _Types...>::_IsAnyOf; };
template <typename _Ty1, typename ..._Types>
struct _Impl_Dragonian_Lib_Is_Any_Of<_Ty1, _Impl_Dragonian_Lib_Va_List<_Types...>> { constexpr static bool _IsAnyOf = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of<_Ty1, _Types...>::_IsAnyOf; };
template <typename _Ty1>
struct _Impl_Dragonian_Lib_Is_Any_Of<_Ty1> { constexpr static bool _IsAnyOf = false; };
template <typename _Ty1, typename ..._Types>
constexpr bool _Impl_Dragonian_Lib_Is_Any_Of_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of<_Ty1, _Types...>::_IsAnyOf;

template <bool _Condition, typename _Type>
struct _Impl_Dragonian_Lib_Enable_If {};
template <typename _Type>
struct _Impl_Dragonian_Lib_Enable_If<true, _Type> { using _DeclType = _Type; };
template <bool _Condition, typename _Type = void>
using _Impl_Dragonian_Lib_Enable_If_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Enable_If<_Condition, _Type>::_DeclType;

template <typename _Type>
struct _Impl_Dragonian_Lib_Is_LValue_Reference { constexpr static bool _IsLValueReference = false; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_LValue_Reference<_Type&> { constexpr static bool _IsLValueReference = true; };
template <typename _Type>
constexpr bool _Impl_Dragonian_Lib_Is_LValue_Reference_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_LValue_Reference<_Type>::_IsLValueReference;

template <typename _Type>
struct _Impl_Dragonian_Lib_Is_RValue_Reference { constexpr static bool _IsRValueReference = false; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_RValue_Reference<_Type&&> { constexpr static bool _IsRValueReference = true; };
template <typename _Type>
constexpr bool _Impl_Dragonian_Lib_Is_RValue_Reference_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_RValue_Reference<_Type>::_IsRValueReference;

template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Reference { constexpr static bool _IsReference = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_LValue_Reference_v<_Type> || _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_RValue_Reference_v<_Type>; };
template <typename _Type>
constexpr bool _Impl_Dragonian_Lib_Is_Reference_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Reference<_Type>::_IsReference;

template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Pointer { constexpr static bool _IsPointer = false; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Pointer<_Type*> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Pointer<_Type* const> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Pointer<_Type* volatile> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Is_Pointer<_Type* const volatile> { constexpr static bool _IsPointer = true; };
template <typename _Type>
constexpr bool _Impl_Dragonian_Lib_Is_Pointer_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Pointer<_Type>::_IsPointer;

struct _Impl_Dragonian_Lib_Always_False_Struct;
template <typename _Type>
struct _Impl_Dragonian_Lib_Always_False
{
	bool Value = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Impl_Dragonian_Lib_Always_False_Struct, _Type>;
};
template <typename _Type>
constexpr bool _Impl_Dragonian_Lib_Always_False_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Always_False<_Type>::value;

template <typename _Type>
constexpr _Type _Impl_Dragonian_Lib_Instance_Of()
{
	throw std::exception("Invalid Instance Of!");
}

template <typename _TypeDst, typename _TypeSrc>
class _Impl_Dragonian_Lib_Could_Be_Converted_From_Class
{
	template <typename _SrcT>
	static constexpr auto Check(int) -> decltype(_TypeDst(_Impl_Dragonian_Lib_Instance_Of<_SrcT>()), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _Condition = decltype(Check<_TypeSrc>(0))::value;
};
template <typename _TypeDst, typename _TypeSrc>
constexpr auto _Impl_Dragonian_Lib_Could_Be_Converted_From_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Could_Be_Converted_From_Class<_TypeDst, _TypeSrc>::_Condition;
template <typename _TypeSrc, typename _TypeDst>
constexpr auto _Impl_Dragonian_Lib_Could_Be_Converted_To_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Could_Be_Converted_From_v<_TypeDst, _TypeSrc>;

template<bool _Condition1, bool _Condition2>
constexpr auto _Impl_Dragonian_Lib_And_v = _Condition1 && _Condition2;

//***********************************************Callable TypeDecl***************************************************//
template<typename _Callable, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Callable_Return
{
	using Type = decltype(
		_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Instance_Of<_Callable>()(
			_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Instance_Of<_ArgTypes>()...
			)
		);
};
template<typename _Callable, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Callable_Return<_Callable, _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Va_List<_ArgTypes...>>
{
	using Type = decltype(
		_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Instance_Of<_Callable>()(
			_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Instance_Of<_ArgTypes>()...
			)
		);
};
template<typename _Callable, typename ..._ArgTypes>
using _Impl_Dragonian_Lib_Callable_Return_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Callable_Return<_Callable, _ArgTypes...>::Type;

//*************************************************Modification****************************************************//
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Pointer { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Pointer<_Type*> { using _DeclType = _Type; };
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_Pointer_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Pointer<_Type>::_DeclType;

template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Reference { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Reference<_Type&> { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Reference<_Type&&> { using _DeclType = _Type; };
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_Reference_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Reference<_Type>::_DeclType;

template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Volatile { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Volatile<volatile _Type> { using _DeclType = _Type; };
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_Volatile_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Volatile<_Type>::_DeclType;

template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Const { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_Const<const _Type> { using _DeclType = _Type; };
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_Const_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Const<_Type>::_DeclType;

template <typename _Type>
struct _Impl_Dragonian_Lib_Reference
{
	using _SrcType = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Reference_t<_Type>;
	using _LReference = _SrcType&;
	using _RReference = _SrcType&&;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Lvalue_Reference_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Reference<_Type>::_LReference;
template <typename _Type>
using _Impl_Dragonian_Lib_Rvalue_Reference_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Reference<_Type>::_RReference;

template <typename _Type>
struct _Impl_Dragonian_Lib_Add_Pointer
{
	using _Pointer = _Type*;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Add_Pointer_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Add_Pointer<_Type>::_Pointer;

template <typename _Type>
struct _Impl_Dragonian_Lib_Pointer
{
	using _SrcType = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Pointer_t<_Type>;
	using _Pointer = _SrcType*;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Pointer_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Pointer<_Type>::_Pointer;

template <typename _Type>
struct _Impl_Dragonian_Lib_Add_Const
{
	using _SrcType = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Const_t<_Type>;
	using _Const = const _SrcType;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Const_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Add_Const<_Type>::_Const;

template <typename _Type>
struct _Impl_Dragonian_Lib_Add_Volatile
{
	using _SrcType = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Volatile_t<_Type>;
	using _Volatile = volatile _SrcType;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Volatile_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Add_Volatile<_Type>::_Volatile;

template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_CV
{
	using _DeclType = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Const_t<_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_Volatile_t<_Type>>;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_CV_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_CV<_Type>::_DeclType;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV { using _DeclType = _Type; };
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <const _Type>;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <volatile _Type>;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type*>;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type&>;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type&&>;
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <const _Type>
{
	using _DeclType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;
};
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <volatile _Type>
{
	using _DeclType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;
};
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type*>
{
	using _DeclType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;
};
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type&>
{
	using _DeclType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;
};
template <typename _Type>
struct _Impl_Dragonian_Lib_Remove_ARPCV <_Type&&>
{
	using _DeclType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Remove_ARPCV_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV<_Type>::_DeclType;

//***********************************************Decl Callable Type************************************************//
template <typename Type>
class _Impl_Dragonian_Lib_Is_Callable_Object
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(&Objty::operator(), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _IsCallable = decltype(Check<Type>(0))::value;
};
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Is_Callable_Object_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Callable_Object<Type>::_IsCallable;

template<typename Objt>
struct _Impl_Dragonian_Lib_Is_Callable { constexpr static bool _IsCallable = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Callable_Object<Objt>::_IsCallable; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(*)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(*&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(*&&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_Is_Callable<_Ret(&&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
constexpr bool _Impl_Dragonian_Lib_Is_Callable_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Callable<_Ret>::_IsCallable;

template<typename Objt>
struct _Impl_Dragonian_Lib_In_Decl_Callable
{
	static_assert(_D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Callable_Object<Objt>::_IsCallable, "Type is not callable!");
	using _Callable = decltype(Objt::operator());
};
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_In_Decl_Callable<_Ret(_ArgTypes...)>
{
	using _Callable = _Ret(_ArgTypes...);
	using _ReturnType = _Ret;
	using _ArgumentTypes = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Va_List<_ArgTypes...>;
};
template<typename _Ret, typename ..._ArgTypes>
struct _Impl_Dragonian_Lib_In_Decl_Callable<_Ret(_ArgTypes...) const>
{
	using _Callable = _Ret(_ArgTypes...);
	using _ReturnType = _Ret;
	using _ArgumentTypes = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Va_List<_ArgTypes...>;
};
template<typename Objt>
struct _Impl_Dragonian_Lib_Decl_Callable
{
	using _Obj = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV_t<Objt>;
	using _Decl_Callable = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_In_Decl_Callable<_Obj>;
	using _Callable = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Remove_ARPCV_t<typename _Decl_Callable::_Callable>;
	using _ReturnType = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_In_Decl_Callable<_Callable>::_ReturnType;
	using _ArgumentTypes = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_In_Decl_Callable<_Callable>::_ArgumentTypes;
};
template<typename Objt>
using _Impl_Dragonian_Lib_Callable_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Decl_Callable<Objt>::_Callable;
template<typename Objt>
using _Impl_Dragonian_Lib_Return_Type_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Decl_Callable<Objt>::_ReturnType;
template<typename Objt>
using _Impl_Dragonian_Lib_Argument_Types_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Decl_Callable<Objt>::_ArgumentTypes;

//************************************************Calculate Index**************************************************//
template <int64_t Idx, int64_t Range>
struct _Impl_Dragonian_Lib_Calculate_Index
{
	static_assert((Idx < 0 && Idx >= -Range) || (Idx >= 0 && Idx < Range));
	constexpr static int64_t Index = Idx < 0 ? (Range + Idx) : Idx;
};
template <int64_t Idx, int64_t Range>
constexpr int64_t _Impl_Dragonian_Lib_Calculate_Index_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Calculate_Index<Idx, Range>::Index;

//***********************************************Get Va List Type**************************************************//
template <int64_t Idx, typename First, typename ...Rest>
struct _Impl_Dragonian_Lib_Get_Va_List_Type_Inl {};
template <typename First, typename ...Rest>
struct _Impl_Dragonian_Lib_Get_Va_List_Type_Inl<0, First, Rest...> { using _Type = First; };
template <int64_t Idx, typename First, typename ...Rest>
struct _Impl_Dragonian_Lib_Get_Va_List_Type_Inl<Idx, First, Rest...> { using _Type = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Va_List_Type_Inl<Idx - 1, Rest...>::_Type; };
template <int64_t Idx, typename ...Types>
struct _Impl_Dragonian_Lib_Get_Va_List_Type
{
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Index = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Calculate_Index_v<Idx, Size>;
	using Type = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Va_List_Type_Inl<Index, Types...>::_Type;
};
template <int64_t Idx, typename ...Types>
struct _Impl_Dragonian_Lib_Get_Va_List_Type<Idx, _Impl_Dragonian_Lib_Va_List<Types...>>
{
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Index = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Calculate_Index_v<Idx, Size>;
	using Type = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Va_List_Type_Inl<Index, Types...>::_Type;
};
template <int64_t Idx, typename ...Types>
using _Impl_Dragonian_Lib_Get_Va_List_Type_t = typename _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Va_List_Type<Idx, Types...>::Type;

//***********************************************Get Value At Index*************************************************//
template <int64_t Index, typename First, typename... Rest>
struct _Impl_Dragonian_Lib_Get_Value_At_Index {};
template <int64_t Index, typename First, typename... Rest>
struct _Impl_Dragonian_Lib_Get_Value_At_Index<Index, First, Rest...> {
	static constexpr auto Get(First, Rest... rest) {
		return _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Value_At_Index<Index - 1, Rest...>::Get(rest...);
	}
};
template <typename First, typename... Rest>
struct _Impl_Dragonian_Lib_Get_Value_At_Index<0, First, Rest...> {
	static constexpr First Get(First first, Rest...) {
		return first;
	}
};
template <int64_t Index, typename... Types>
struct _Impl_Dragonian_Lib_Get_Value_At_Index<Index, _Impl_Dragonian_Lib_Va_List<Types...>> {
	static constexpr auto Get(Types... rest) {
		return _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Value_At_Index<Index - 1, Types...>::Get(rest...);
	}
};
template <int64_t Index, typename... Types>
constexpr auto _Impl_Dragonian_Lib_Get_Value_At_Index_v(Types... args) {
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Idx = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Calculate_Index_v<Index, Size>;
	return _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Get_Value_At_Index<Idx, Types...>::Get(args...);
}

//***********************************************Is Arithmetic Type************************************************//

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Floating_Point_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, float, double, long double, Float16, Float8, BFloat16>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Integer_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Signed_Integer_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, bool, Int8, Int16, Int32, Int64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Unsigned_Integer_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, UInt8, UInt16, UInt32, UInt64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Complex_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, Complex32, Complex64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Bool_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, bool>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Arithmetic_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Floating_Point_v<_Type> || _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Integer_v<_Type> || _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Complex_v<_Type> || _Impl_Dragonian_Lib_Is_Bool_v<_Type>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Avx256_Supported_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Avx256_Supported_Floating_Point_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, Float32, Float64>;

template <typename _Type>
constexpr auto _Impl_Dragonian_Lib_Is_Avx256_Supported_Integer_v = _D_Dragonian_Lib_Namespace _Impl_Dragonian_Lib_Is_Any_Of_v<_Impl_Dragonian_Lib_Remove_CV_t<_Type>, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64>;

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

private:
	FILE* file_ = nullptr;
};
_D_Dragonian_Lib_Space_End
