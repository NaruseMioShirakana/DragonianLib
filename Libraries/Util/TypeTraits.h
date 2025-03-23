﻿/**
 * @file TypeTraits.h
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
 * @brief Type traits for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TypeDef.h"

#define _D_Dragonian_Lib_Type_Traits_Namespace_Begin _D_Dragonian_Lib_Space_Begin namespace TypeTraits {
#define _D_Dragonian_Lib_Type_Traits_Namespace_End } _D_Dragonian_Lib_Space_End
#define _D_Dragonian_Lib_Type_Traits_Namespace ::DragonianLib::TypeTraits::

_D_Dragonian_Lib_Type_Traits_Namespace_Begin

// ReSharper disable all

//*********************************** Reference Pointer Const Volatile ************************************//
template <typename _Type>
struct RemovePointer { using _MyType = _Type; };
template <typename _Type>
struct RemovePointer<_Type*> { using _MyType = _Type; };
template <typename _Type>
using RemovePointerType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemovePointer<_Type>::_MyType;

template <typename _Type>
struct RemoveReference { using _MyType = _Type; };
template <typename _Type>
struct RemoveReference<_Type&> { using _MyType = _Type; };
template <typename _Type>
struct RemoveReference<_Type&&> { using _MyType = _Type; };
template <typename _Type>
using RemoveReferenceType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveReference<_Type>::_MyType;

template <typename _Type>
struct RemoveVolatile { using _MyType = _Type; };
template <typename _Type>
struct RemoveVolatile<volatile _Type> { using _MyType = _Type; };
template <typename _Type>
using RemoveVolatileType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveVolatile<_Type>::_MyType;

template <typename _Type>
struct RemoveConst { using _MyType = _Type; };
template <typename _Type>
struct RemoveConst<const _Type> { using _MyType = _Type; };
template <typename _Type>
using RemoveConstType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveConst<_Type>::_MyType;

template <typename _Type>
struct Reference
{
	using _SrcType = _D_Dragonian_Lib_Type_Traits_Namespace RemoveReferenceType<_Type>;
	using _LReference = _SrcType&;
	using _RReference = _SrcType&&;
};
template <typename _Type>
using LReferenceType = typename _D_Dragonian_Lib_Type_Traits_Namespace Reference<_Type>::_LReference;
template <typename _Type>
using RReferenceType = typename _D_Dragonian_Lib_Type_Traits_Namespace Reference<_Type>::_RReference;

template <typename _Type>
struct AddPointer
{
	using _Pointer = _Type*;
};
template <typename _Type>
using AddPointerType = typename _D_Dragonian_Lib_Type_Traits_Namespace AddPointer<_Type>::_Pointer;

template <typename _Type>
struct Pointer
{
	using _SrcType = _D_Dragonian_Lib_Type_Traits_Namespace RemovePointerType<_Type>;
	using _Pointer = _SrcType*;
};
template <typename _Type>
using PointerType = typename _D_Dragonian_Lib_Type_Traits_Namespace Pointer<_Type>::_Pointer;

template <typename _Type>
struct AddConst
{
	using _SrcType = _D_Dragonian_Lib_Type_Traits_Namespace RemoveConstType<_Type>;
	using _Const = const _SrcType;
};
template <typename _Type>
using ConstType = typename _D_Dragonian_Lib_Type_Traits_Namespace AddConst<_Type>::_Const;

template <typename _Type>
struct AddVolatile
{
	using _SrcType = _D_Dragonian_Lib_Type_Traits_Namespace RemoveVolatileType<_Type>;
	using _Volatile = volatile _SrcType;
};
template <typename _Type>
using VolatileType = typename _D_Dragonian_Lib_Type_Traits_Namespace AddVolatile<_Type>::_Volatile;

template <typename _Type>
struct RemoveCV
{
	using _MyType = _D_Dragonian_Lib_Type_Traits_Namespace RemoveConstType<_D_Dragonian_Lib_Type_Traits_Namespace RemoveVolatileType<_Type>>;
};
template <typename _Type>
using RemoveCVType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveCV<_Type>::_MyType;
template <typename _Type>
struct RemoveARPCV { using _MyType = _Type; };
template <typename _Type>
struct RemoveARPCV <const _Type>;
template <typename _Type>
struct RemoveARPCV <volatile _Type>;
template <typename _Type>
struct RemoveARPCV <_Type*>;
template <typename _Type>
struct RemoveARPCV <_Type&>;
template <typename _Type>
struct RemoveARPCV <_Type&&>;
template <typename _Type>
struct RemoveARPCV <const _Type>
{
	using _MyType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;
};
template <typename _Type>
struct RemoveARPCV <volatile _Type>
{
	using _MyType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;
};
template <typename _Type>
struct RemoveARPCV <_Type*>
{
	using _MyType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;
};
template <typename _Type>
struct RemoveARPCV <_Type&>
{
	using _MyType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;
};
template <typename _Type>
struct RemoveARPCV <_Type&&>
{
	using _MyType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;
};
template <typename _Type>
using RemoveARPCVType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCV<_Type>::_MyType;

//********************************************** Conditional **********************************************//
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
struct Conditional;
template <typename _Tyt, typename _Tyf>
struct Conditional<true, _Tyt, _Tyf> { using _Mytype = _Tyt; };
template <typename _Tyt, typename _Tyf>
struct Conditional<false, _Tyt, _Tyf> { using _Mytype = _Tyf; };
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
using ConditionalType = typename _D_Dragonian_Lib_Type_Traits_Namespace Conditional<_Test, _Tyt, _Tyf>::_Mytype;
template <bool _Test, typename _Tyt = void, typename _Tyf = void>
using ConditionalType = _D_Dragonian_Lib_Type_Traits_Namespace ConditionalType<_Test, _Tyt, _Tyf>;

template <typename _Ty1, typename _Ty2>
struct IsSameType
{
	constexpr static bool _IsSame = false;
};
template <typename _Ty1>
struct IsSameType<_Ty1, _Ty1>
{
	constexpr static bool _IsSame = true;
};
template <typename _Ty1, typename _Ty2>
constexpr bool IsSameTypeValue = _D_Dragonian_Lib_Type_Traits_Namespace IsSameType<_Ty1, _Ty2>::_IsSame;

template <typename _Ty1, typename ..._Types>
struct IsAnyOf { constexpr static bool _IsAnyOf = false; };
template <typename _Ty1, typename _Ty2, typename ..._Types>
struct IsAnyOf<_Ty1, _Ty2, _Types...> { constexpr static bool _IsAnyOf = _D_Dragonian_Lib_Type_Traits_Namespace IsSameTypeValue<_Ty1, _Ty2> || _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOf<_Ty1, _Types...>::_IsAnyOf; };
template <typename _Ty1, typename ..._Types>
struct IsAnyOf<_Ty1, _D_Dragonian_Lib_Namespace GeneralizedList<_Types...>> { constexpr static bool _IsAnyOf = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOf<_Ty1, _Types...>::_IsAnyOf; };
template <typename _Ty1>
struct IsAnyOf<_Ty1> { constexpr static bool _IsAnyOf = false; };
template <typename _Ty1, typename ..._Types>
constexpr bool IsAnyOfValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOf<_Ty1, _Types...>::_IsAnyOf;

template <bool _Condition, typename _Type>
struct EnableIf {};
template <typename _Type>
struct EnableIf<true, _Type> { using _Mytype = _Type; };
template <bool _Condition, typename _Type = void>
using EnableIfType = typename _D_Dragonian_Lib_Type_Traits_Namespace EnableIf<_Condition, _Type>::_Mytype;

template <typename _Type>
struct IsLReference { constexpr static bool _IsLValueReference = false; };
template <typename _Type>
struct IsLReference<_Type&> { constexpr static bool _IsLValueReference = true; };
template <typename _Type>
constexpr bool IsLReferenceValue = _D_Dragonian_Lib_Type_Traits_Namespace IsLReference<_Type>::_IsLValueReference;

template <typename _Type>
struct IsRReference { constexpr static bool _IsRValueReference = false; };
template <typename _Type>
struct IsRReference<_Type&&> { constexpr static bool _IsRValueReference = true; };
template <typename _Type>
constexpr bool IsRReferenceValue = _D_Dragonian_Lib_Type_Traits_Namespace IsRReference<_Type>::_IsRValueReference;

template <typename _Type>
struct IsReference { constexpr static bool _IsReference = _D_Dragonian_Lib_Type_Traits_Namespace IsLReferenceValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsRReferenceValue<_Type>; };
template <typename _Type>
constexpr bool IsReferenceValue = _D_Dragonian_Lib_Type_Traits_Namespace IsReference<_Type>::_IsReference;

template <typename _Type>
struct IsPointer { constexpr static bool _IsPointer = false; };
template <typename _Type>
struct IsPointer<_Type*> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct IsPointer<_Type* const> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct IsPointer<_Type* volatile> { constexpr static bool _IsPointer = true; };
template <typename _Type>
struct IsPointer<_Type* const volatile> { constexpr static bool _IsPointer = true; };
template <typename _Type>
constexpr bool IsPointerValue = _D_Dragonian_Lib_Type_Traits_Namespace IsPointer<_Type>::_IsPointer;

template <typename _Type>
struct IsConst { constexpr static bool _IsConst = false; };
template <typename _Type>
struct IsConst<const _Type> { constexpr static bool _IsConst = true; };
template <typename _Type>
constexpr bool IsConstValue = _D_Dragonian_Lib_Type_Traits_Namespace IsConst<_Type>::_IsConst;

template <typename _Type>
struct IsVolatile { constexpr static bool _IsVolatile = false; };
template <typename _Type>
struct IsVolatile<volatile _Type> { constexpr static bool _IsVolatile = true; };
template <typename _Type>
constexpr bool IsVolatileValue = _D_Dragonian_Lib_Type_Traits_Namespace IsVolatile<_Type>::_IsVolatile;

template <typename _Type>
struct IsARPCV { constexpr static bool _IsARPCV = _D_Dragonian_Lib_Type_Traits_Namespace IsConstValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsVolatileValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsPointerValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsReferenceValue<_Type>; };
template <typename _Type>
constexpr bool IsARPCVValue = _D_Dragonian_Lib_Type_Traits_Namespace IsARPCV<_Type>::_IsARPCV;

struct AlwaysFalseStruct;
template <typename _Type>
struct AlwaysFalse
{
	static constexpr bool Value = _D_Dragonian_Lib_Type_Traits_Namespace IsSameTypeValue<AlwaysFalseStruct, _Type>;
};
template <typename _Type>
constexpr bool AlwaysFalseValue = _D_Dragonian_Lib_Type_Traits_Namespace AlwaysFalse<_Type>::Value;

template <typename _Type>
constexpr LReferenceType<_Type> InstanceOf()
{
	static_assert(false, "Calling InstanceOf() is illegal!");
}

template <typename _TypeDst, typename _TypeSrc>
class CouldBeConvertedFromClass
{
	template <typename _SrcT>
	static constexpr auto Check(int) -> decltype(_TypeDst(InstanceOf<_SrcT>()), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _Condition = decltype(Check<_TypeSrc>(0))::value;
};
template <typename _TypeDst, typename _TypeSrc>
constexpr auto CouldBeConvertedFromValue = _D_Dragonian_Lib_Type_Traits_Namespace CouldBeConvertedFromClass<_TypeDst, _TypeSrc>::_Condition;
template <typename _TypeSrc, typename _TypeDst>
constexpr auto CouldBeConvertedToValue = _D_Dragonian_Lib_Type_Traits_Namespace CouldBeConvertedFromValue<_TypeDst, _TypeSrc>;

template<bool _Condition1, bool _Condition2>
constexpr auto AndValue = _Condition1 && _Condition2;

template <typename Type>
class IsCallableObject
{
	template <typename Objty>
	static constexpr auto Check(int) -> decltype(&Objty::operator(), std::true_type()) { return{}; }
	template <typename>
	static constexpr std::false_type Check(...) { return{}; }
public:
	static constexpr bool _IsCallable = decltype(Check<Type>(0))::value;
};
template <typename Type>
constexpr bool IsCallableObjectValue = _D_Dragonian_Lib_Type_Traits_Namespace IsCallableObject<Type>::_IsCallable;

template<typename Objt>
struct IsCallable { constexpr static bool _IsCallable = _D_Dragonian_Lib_Type_Traits_Namespace IsCallableObject<Objt>::_IsCallable; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(*)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(*&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(*&&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename _Ret, typename ..._ArgTypes>
struct IsCallable<_Ret(&&)(_ArgTypes...)> { constexpr static bool _IsCallable = true; };
template<typename Objt>
constexpr bool IsCallableValue = _D_Dragonian_Lib_Type_Traits_Namespace IsCallable<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<Objt>>::_IsCallable;

template <typename _Type>
struct RemoveNoexcept { using _MyType = _Type; };
template <typename _Ret, typename ..._ArgTypes>
struct RemoveNoexcept<_Ret(_ArgTypes...) noexcept> { using _MyType = _Ret(_ArgTypes...); };
template <typename _Ret, typename ..._ArgTypes>
struct RemoveNoexcept<_Ret(_ArgTypes...) const noexcept> { using _MyType = _Ret(_ArgTypes...) const; };
template <typename _Ret, typename ..._ArgTypes>
struct RemoveNoexcept<_Ret(* const)(_ArgTypes...) noexcept> { using _MyType = _Ret(* const)(_ArgTypes...); };
template <typename _Type>
using RemoveNoexceptType = typename _D_Dragonian_Lib_Type_Traits_Namespace RemoveNoexcept<_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>::_MyType;

template<typename Objt>
struct ExtractCallableTypeInfo
{
	static_assert(_D_Dragonian_Lib_Type_Traits_Namespace IsCallableObject<Objt>::_IsCallable, "Type is not callable!");
	using _Callable = decltype(Objt::operator());
};
template<typename _Ret, typename ..._ArgTypes>
struct ExtractCallableTypeInfo<_Ret(_ArgTypes...)>
{
	using _Callable = _Ret(_ArgTypes...);
	using _ReturnType = _Ret;
	using _ArgumentTypes = _D_Dragonian_Lib_Namespace GeneralizedList<_ArgTypes...>;
};
template<typename _Ret, typename ..._ArgTypes>
struct ExtractCallableTypeInfo<_Ret(_ArgTypes...) const>
{
	using _Callable = _Ret(_ArgTypes...);
	using _ReturnType = _Ret;
	using _ArgumentTypes = _D_Dragonian_Lib_Namespace GeneralizedList<_ArgTypes...>;
};
template<typename Objt>
struct ExtractCallableInfo
{
	using _Obj = _D_Dragonian_Lib_Type_Traits_Namespace RemoveNoexceptType<_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<Objt>>;
	using _My_Callable = _D_Dragonian_Lib_Type_Traits_Namespace ExtractCallableTypeInfo<_Obj>;
	using _Callable = _D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<typename _My_Callable::_Callable>;
	using _ReturnType = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractCallableTypeInfo<_Callable>::_ReturnType;
	using _ArgumentTypes = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractCallableTypeInfo<_Callable>::_ArgumentTypes;
};
template<typename Objt>
using CallableType = typename _D_Dragonian_Lib_Type_Traits_Namespace
ExtractCallableInfo<Objt>::_Callable;
template<typename Objt>
using ReturnTypeType = typename _D_Dragonian_Lib_Type_Traits_Namespace
ExtractCallableInfo<Objt>::_ReturnType;
template<typename Objt>
using ArgumentTypesType = typename _D_Dragonian_Lib_Type_Traits_Namespace
ExtractCallableInfo<Objt>::_ArgumentTypes;

template<typename FunType, typename ...ArgTypes>
concept IsInvokeableValue = requires(FunType && fn, ArgTypes&&... args)
{
	{ std::forward<FunType>(fn)(std::forward<ArgTypes>(args)...) }; // Check if the function is callable
};

template <int64_t Idx, int64_t Range>
struct CalculateIndex
{
	static_assert((Idx < 0 && Idx >= -Range) || (Idx >= 0 && Idx < Range));
	constexpr static int64_t Index = Idx < 0 ? (Range + Idx) : Idx;
};
template <int64_t Idx, int64_t Range>
constexpr int64_t CalculateIndexValue = _D_Dragonian_Lib_Type_Traits_Namespace CalculateIndex<Idx, Range>::Index;

template <int64_t Idx, typename First, typename ...Rest>
struct GetVaListTypeAtIndex;
template <typename First, typename ...Rest>
struct GetVaListTypeAtIndex<0, First, Rest...> { using _Type = First; };
template <int64_t Idx, typename First, typename ...Rest>
struct GetVaListTypeAtIndex<Idx, First, Rest...> { using _Type = typename _D_Dragonian_Lib_Type_Traits_Namespace GetVaListTypeAtIndex<Idx - 1, Rest...>::_Type; };
template <int64_t Idx, typename ...Types>
struct GetVaListTypeAt
{
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Index = _D_Dragonian_Lib_Type_Traits_Namespace CalculateIndexValue<Idx, Size>;
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace GetVaListTypeAtIndex<Index, Types...>::_Type;
};
template <int64_t Idx, typename ...Types>
struct GetVaListTypeAt<Idx, _D_Dragonian_Lib_Namespace GeneralizedList<Types...>>
{
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Index = _D_Dragonian_Lib_Type_Traits_Namespace CalculateIndexValue<Idx, Size>;
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace GetVaListTypeAtIndex<Index, Types...>::_Type;
};
template <int64_t Idx, typename ...Types>
using GetVaListTypeAtType = typename _D_Dragonian_Lib_Type_Traits_Namespace GetVaListTypeAt<Idx, Types...>::Type;

template <int64_t Index, typename First, typename... Rest>
struct GetValueAt;
template <int64_t Index, typename First, typename... Rest>
struct GetValueAt<Index, First, Rest...> {
	static constexpr decltype(auto) Get(First&&, Rest&&... rest) {
		return _D_Dragonian_Lib_Type_Traits_Namespace GetValueAt<Index - 1, Rest...>::Get(std::forward<Rest>(rest)...);
	}
};
template <typename First, typename... Rest>
struct GetValueAt<0, First, Rest...> {
	static constexpr decltype(auto) Get(First&& first, Rest&&...) {
		return std::forward<First>(first);
	}
};
template <int64_t Index, typename... Types>
struct GetValueAt<Index, _D_Dragonian_Lib_Namespace GeneralizedList<Types...>> {
	static constexpr decltype(auto) Get(Types&&... args) {
		return _D_Dragonian_Lib_Type_Traits_Namespace GetValueAt<Index, Types...>::Get(std::forward<Types>(args)...);
	}
};
template <int64_t Index, typename... Types>
constexpr decltype(auto) GetValue(Types&&... args) {
	constexpr static auto Size = sizeof...(Types);
	constexpr static auto Idx = _D_Dragonian_Lib_Type_Traits_Namespace CalculateIndexValue<Index, Size>;
	return _D_Dragonian_Lib_Type_Traits_Namespace GetValueAt<Idx, Types...>::Get(std::forward<Types>(args)...);
}

template <typename _Type>
constexpr auto IsFloatingPointValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue <RemoveCVType<_Type>, Float32, Float64, long double, Float16, BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ>;
template <typename _Type>
constexpr auto IsIntegerValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64>;
template <typename _Type>
constexpr auto IsSignedIntegerValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Boolean, Int8, Int16, Int32, Int64>;
template <typename _Type>
constexpr auto IsUnsignedIntegerValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, UInt8, UInt16, UInt32, UInt64>;
template <typename _Type>
constexpr auto IsComplexValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Complex32, Complex64>;
template <typename _Type>
constexpr auto IsBoolValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Boolean>;
template <typename _Type>
constexpr auto IsArithmeticValue = _D_Dragonian_Lib_Type_Traits_Namespace IsFloatingPointValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsIntegerValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsComplexValue<_Type> || IsBoolValue<_Type>;
template <typename _Type>
constexpr auto IsAvx256SupportedValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Complex32, Complex64, Boolean>;
template <typename _Type>
constexpr auto IsAvx256SupportedFloatingPointValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Float32, Float64>;
template <typename _Type>
constexpr auto IsAvx256SupportedIntegerValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64>;
template <typename _Type>
constexpr auto IsCppStringValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, std::string, std::wstring>;
template <typename _Type>
constexpr auto IsCStringValue = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<RemoveCVType<_Type>, const char*, const wchar_t*>;
template <typename _Type>
constexpr auto IsStringValue = _D_Dragonian_Lib_Type_Traits_Namespace IsCppStringValue<_Type> || _D_Dragonian_Lib_Type_Traits_Namespace IsCStringValue<_Type>;

template <typename _Type>
constexpr auto ExtractRank = 0;
template <typename _Type>
constexpr auto ExtractRank<_Type[]> = 1 + _D_Dragonian_Lib_Type_Traits_Namespace ExtractRank<_Type>;
template <typename _Type, size_t _Size>
constexpr auto ExtractRank<_Type[_Size]> = 1 + _D_Dragonian_Lib_Type_Traits_Namespace ExtractRank<_Type>;
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
constexpr auto ExtractRank<_ObjType<_ValueType, _ValueSize>> = 1 + _D_Dragonian_Lib_Type_Traits_Namespace ExtractRank<_ValueType>;
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
constexpr auto ExtractRank<_ObjType<_ValueSize, _ValueType>> = 1 + _D_Dragonian_Lib_Type_Traits_Namespace ExtractRank<_ValueType>;
template <typename _Type>
constexpr auto ExtractRankValue = _D_Dragonian_Lib_Type_Traits_Namespace ExtractRank<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type>
constexpr auto ExtractInitializerListRank = 0;
template <typename _Type>
constexpr auto ExtractInitializerListRank<std::initializer_list<_Type>> = 1 + _D_Dragonian_Lib_Type_Traits_Namespace ExtractInitializerListRank<_Type>;
template <typename _Type>
constexpr auto ExtractInitializerListRankValue = _D_Dragonian_Lib_Type_Traits_Namespace ExtractInitializerListRank<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type>
constexpr bool IsArray = false;
template <typename _Type, size_t _Size>
constexpr bool IsArray<_Type[_Size]> = true;
template <typename _Type>
constexpr bool IsArray<_Type[]> = true;
template <typename _Type>
constexpr bool IsArrayValue = _D_Dragonian_Lib_Type_Traits_Namespace IsArray<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type>
constexpr bool IsArrayLike = false;
template <typename _Type>
constexpr bool IsArrayLike<_Type[]> = true;
template <typename _Type, size_t _Size>
constexpr bool IsArrayLike<_Type[_Size]> = true;
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
constexpr bool IsArrayLike<_ObjType<_ValueType, _ValueSize>> = true;
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
constexpr bool IsArrayLike<_ObjType<_ValueSize, _ValueType>> = true;
template <typename _Type>
constexpr bool IsArrayLikeValue = _D_Dragonian_Lib_Type_Traits_Namespace IsArrayLike<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type>
constexpr bool IsInitializerList = false;
template <typename _Type>
constexpr bool IsInitializerList<std::initializer_list<_Type>> = true;
template <typename _Type>
constexpr bool IsInitializerListValue = _D_Dragonian_Lib_Type_Traits_Namespace IsInitializerList<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type>
constexpr auto IsArrayLikeOrInitializerList = _D_Dragonian_Lib_Type_Traits_Namespace IsArrayLikeValue<_Type> ||
_D_Dragonian_Lib_Type_Traits_Namespace IsInitializerListValue<_Type>;

template <typename _Type>
struct ExtractTypeOfArray;
template <typename _Type, size_t _Size>
struct ExtractTypeOfArray<_Type[_Size]>
{
	using Type = _Type;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractTypeOfArray<_ObjType<_ValueType, _ValueSize>>
{
	using Type = _ValueType;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractTypeOfArray<_ObjType<_ValueSize, _ValueType>>
{
	using Type = _ValueType;
};
template <typename _Type>
using ExtractTypeOfArrayType = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArray<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>::Type;

template <typename _Type>
struct ExtractInnerTypeOfArray
{
	using Type = _Type;
};
template <typename _Type, size_t _Size>
struct ExtractInnerTypeOfArray<_Type[_Size]>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractInnerTypeOfArray<_Type>::Type;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractInnerTypeOfArray<_ObjType<_ValueType, _ValueSize>>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractInnerTypeOfArray<_ValueType>::Type;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractInnerTypeOfArray<_ObjType<_ValueSize, _ValueType>>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractInnerTypeOfArray<_ValueType>::Type;
};
template <typename _Type>
using ExtractInnerTypeOfArrayType = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractInnerTypeOfArray<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>::Type;

template <typename _Type, size_t _Index>
struct ExtractTypeOfArrayAt
{
	using Type = _Type;
};
template <typename _Type, size_t _Index>
struct ExtractTypeOfArrayAt<_Type[], _Index>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArrayAt<_Type, _Index - 1>::Type;
};
template <typename _Type, size_t _Size, size_t _Index>
struct ExtractTypeOfArrayAt<_Type[_Size], _Index>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArrayAt<_Type, _Index - 1>::Type;
};
template <typename _Type, size_t _Size>
struct ExtractTypeOfArrayAt<_Type[_Size], 0>
{
	using Type = _Type;
};
template <typename _Type>
struct ExtractTypeOfArrayAt<_Type[], 0>
{
	using Type = _Type;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize, size_t _Index>
struct ExtractTypeOfArrayAt<_ObjType<_ValueType, _ValueSize>, _Index>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArrayAt<_ValueType, _Index - 1>::Type;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType, size_t _Index>
struct ExtractTypeOfArrayAt<_ObjType<_ValueSize, _ValueType>, _Index>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArrayAt<_ValueType, _Index - 1>::Type;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractTypeOfArrayAt<_ObjType<_ValueType, _ValueSize>, 0>
{
	using Type = _ValueType;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractTypeOfArrayAt<_ObjType<_ValueSize, _ValueType>, 0>
{
	using Type = _ValueType;
};
template <typename _Type, size_t _Index>
using ExtractTypeOfArrayAtType = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfArrayAt<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>, _Index>::Type;

template <typename _Type, size_t _Index>
struct ExtractTypeOfInitializerListAt
{
	using Type = _Type;
};
template <typename _Type, size_t _Index>
struct ExtractTypeOfInitializerListAt<std::initializer_list<_Type>, _Index>
{
	using Type = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfInitializerListAt<_Type, _Index - 1>::Type;
};
template <typename _Type>
struct ExtractTypeOfInitializerListAt<std::initializer_list<_Type>, 0>
{
	using Type = _Type;
};
template <typename _Type, size_t _Index>
using ExtractTypeOfInitializerListAtType = typename _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfInitializerListAt<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>, _Index>::Type;

template <typename _Type>
constexpr auto ExtractArraySize = 0;
template <typename _Type, size_t _Size>
constexpr auto ExtractArraySize<_Type[_Size]> = _Size;
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
constexpr auto ExtractArraySize<_ObjType<_ValueType, _ValueSize>> = _ValueSize;
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
constexpr auto ExtractArraySize<_ObjType<_ValueSize, _ValueType>> = _ValueSize;
template <typename _Type>
constexpr auto ExtractArraySizeValue = _D_Dragonian_Lib_Type_Traits_Namespace ExtractArraySize<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>>;

template <typename _Type, size_t _Index>
struct ExtractArraySizeAt;
template <typename _Type, size_t _Index>
struct ExtractArraySizeAt<_Type[], _Index>
{
	static_assert(true, "_Type[] has no size.");
};
template <typename _Type, size_t _Size, size_t _Index>
struct ExtractArraySizeAt<_Type[_Size], _Index>
{
	static constexpr auto Size = _D_Dragonian_Lib_Type_Traits_Namespace ExtractArraySizeAt<_Type, _Index - 1>::Size;
};
template <typename _Type, size_t _Size>
struct ExtractArraySizeAt<_Type[_Size], 0>
{
	static constexpr auto Size = _Size;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize, size_t _Index>
struct ExtractArraySizeAt<_ObjType<_ValueType, _ValueSize>, _Index>
{
	static constexpr auto Size = _D_Dragonian_Lib_Type_Traits_Namespace ExtractArraySizeAt<_ValueType, _Index - 1>::Size;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType, size_t _Index>
struct ExtractArraySizeAt<_ObjType<_ValueSize, _ValueType>, _Index>
{
	static constexpr auto Size = _D_Dragonian_Lib_Type_Traits_Namespace ExtractArraySizeAt<_ValueType, _Index - 1>::Size;
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractArraySizeAt<_ObjType<_ValueType, _ValueSize>, 0>
{
	static constexpr auto Size = _ValueSize;
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractArraySizeAt<_ObjType<_ValueSize, _ValueType>, 0>
{
	static constexpr auto Size = _ValueSize;
};
template <typename _Type, size_t _Index>
constexpr auto ExtractArraySizeAtValue = _D_Dragonian_Lib_Type_Traits_Namespace ExtractArraySizeAt<
	_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>, _Index>::Size;

template <typename _Type>
using ExtractInnerInitializerListType = _D_Dragonian_Lib_Type_Traits_Namespace ExtractTypeOfInitializerListAtType<
	_Type, _D_Dragonian_Lib_Type_Traits_Namespace ExtractInitializerListRankValue<_Type>>;

template <typename _Type>
using InitializerListType = ::DragonianLib::NDInitilizerList<
	ExtractInnerInitializerListType<_Type>, ExtractInitializerListRankValue<_Type>>;

struct IsInvokableWith
{
	template <typename _FunType, typename ..._ArgTypes>
	static constexpr auto CheckConst(const _FunType& _Fun, _ArgTypes... _Args)
		-> decltype(_Fun(_Args...), std::true_type()) {
		return{};
	}

	static constexpr std::false_type CheckConst(...) {
		return{};
	}
};

template <typename _IndexType>
constexpr inline _IndexType BTCalcIndex(_IndexType _Index, _IndexType _Max)
{
	if (_Index < 0)
		_Index += _Max;
	if (_Index >= _Max || _Index < 0)
		return -1;
	return _Index;
}

template <typename _Type, typename ..._ArgTypes>
constexpr auto IsConstructibleValue = std::is_constructible<_Type, _ArgTypes...>::value;

struct AnyConvertible
{
	template <typename ..._SrcType>
	constexpr AnyConvertible(_SrcType&& ..._Src) {}
	template <typename _DstType>
	constexpr operator _DstType& () const;
	template <typename _DstType>
	constexpr operator _DstType && () const;
};

template<typename _Type, size_t N>
constexpr auto MemberCountLowerThan()
{
	return[]<size_t... I>(IndexSequence<I...>)
	{
		return requires{ _Type{ AnyConvertible(I)... }; };
	}(MakeIndexSequence<N>{});
}

template <typename _Type, size_t N = 0>
constexpr size_t MemberCountOf()
{
	if constexpr (
		_D_Dragonian_Lib_Type_Traits_Namespace MemberCountLowerThan<_Type, N>() &&
		!_D_Dragonian_Lib_Type_Traits_Namespace MemberCountLowerThan<_Type, N + 1>()
		)
		return N;
	else
		return _D_Dragonian_Lib_Type_Traits_Namespace MemberCountOf<_Type, N + 1>();
}

template <typename _Type, size_t N1, size_t N2, size_t N3>
constexpr size_t TryPlaceN2AtN1()
{
	return[]<size_t... I>(IndexSequence<I...>)
	{
		return requires{ _Type{ AnyConvertible(I)... }; };
	}(MakeIndexSequence<N1>{});
}

template <typename _Type>
constexpr size_t MemberCountOfValue = _D_Dragonian_Lib_Type_Traits_Namespace MemberCountOf<_D_Dragonian_Lib_Type_Traits_Namespace RemoveARPCVType<_Type>, 0>();

template <typename _Type1, typename _Type2>
concept SameImpl = _D_Dragonian_Lib_Type_Traits_Namespace IsSameTypeValue<_Type1, _Type2>;
template <typename _Type1, typename _Type2>
concept IsType = _D_Dragonian_Lib_Type_Traits_Namespace SameImpl<_Type1, _Type2>&& _D_Dragonian_Lib_Type_Traits_Namespace SameImpl<_Type2, _Type1>;
template <typename _Type1, typename... _Type>
concept IsAnyType = _D_Dragonian_Lib_Type_Traits_Namespace IsAnyOfValue<_Type1, _Type...>;
template <typename _Type1, typename _Type2>
concept NotType = !_D_Dragonian_Lib_Type_Traits_Namespace IsType<_Type1, _Type2>;
template <typename _Type1, typename... _Type>
concept NotAnyType = !_D_Dragonian_Lib_Type_Traits_Namespace IsAnyType<_Type1, _Type...>;

template <typename _Type>
concept CouldIndex = requires(_Type & _Val)
{
	{ _Val[0] };
};

template <typename _Type>
concept HasFrontIncrement = requires(_Type & _Val)
{
	{ ++_Val } -> IsType<_Type&>;
};
template <typename _Type>
concept HasBackIncrement = requires(_Type & _Val)
{
	{ _Val++ } -> IsType<_Type>;
};
template <typename _Type>
concept HasFrontDecrement = requires(_Type & _Val)
{
	{ --_Val } -> IsType<_Type&>;
};
template <typename _Type>
concept HasBackDecrement = requires(_Type & _Val)
{
	{ _Val-- } -> IsType<_Type>;
};

template <typename _Type>
concept HasIntegerAddition = requires(_Type & _Val)
{
	{ _Val + 1 } -> IsType<_Type>;
};
template <typename _Type>
concept HasIntegerSubtraction = requires(_Type & _Val)
{
	{ _Val - 1 } -> IsType<_Type>;
};
template <typename _Type>
concept HasInplaceIntegerAddition = requires(_Type & _Val)
{
	{ _Val += 1 } -> IsType<_Type&>;
};
template <typename _Type>
concept HasInplaceIntegerSubtraction = requires(_Type & _Val)
{
	{ _Val -= 1 } -> IsType<_Type&>;
};

template <typename _Type>
concept HasMemberOperator = requires(_Type & _Val)
{
	{ _Val.operator->() };
};
template <typename _Type>
concept HasUnrefOperator = requires(_Type & _Val)
{
	{ *_Val };
};

template <typename _Type>
concept HasLessOperator = requires(_Type & _Val)
{
	{ _Val < _Val } -> IsType<bool>;
};
template <typename _Type>
concept HasEqualOperator = requires(_Type & _Val)
{
	{ _Val == _Val } -> IsType<bool>;
};

template <typename _Type>
concept IsPointerLike = HasMemberOperator<_Type> && HasUnrefOperator<_Type> && HasFrontIncrement<_Type> && HasBackIncrement<_Type> && HasFrontDecrement<_Type> && HasBackDecrement<_Type> && HasIntegerAddition<_Type> && HasIntegerSubtraction<_Type> && HasInplaceIntegerAddition<_Type> && HasInplaceIntegerSubtraction<_Type> && CouldIndex<_Type> && HasLessOperator<_Type> && HasEqualOperator<_Type>;
template <typename _Type>
concept IsIterator = HasUnrefOperator<_Type> && HasFrontIncrement<_Type> && HasLessOperator<_Type> && HasEqualOperator<_Type>;
template <typename _Type>
concept IsRandomAccessIterator = IsPointerLike<_Type>;

template <typename _Type>
concept HasLRange = requires(_Type & _Val)
{
	{ _Val.begin() } -> IsIterator;
	{ _Val.end() } -> IsIterator;
	{ _Val.end() } -> IsType<decltype(_Val.begin())>;
};
template <typename _Type>
concept HasHRange = requires(_Type & _Val)
{
	{ _Val.Begin() } -> IsIterator;
	{ _Val.End() } -> IsIterator;
	{ _Val.End() } -> IsType<decltype(_Val.Begin())>;
};
template <typename _Type>
concept HasRange = HasLRange<_Type> || HasHRange<_Type>;

_D_Dragonian_Lib_Type_Traits_Namespace_End