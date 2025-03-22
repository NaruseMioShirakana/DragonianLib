#pragma once
#include "../OperatorBase.h"

_D_Dragonian_Lib_Operator_Space_Begin

namespace BinaryOperators
{
	template <typename _Type>
	concept HasClampMaxOp = requires(const _Type & _Val, const _Type & _M) { { _Val > _M }; };
	template <typename _Type>
	concept HasClampMinOp = requires(const _Type & _Val, const _Type & _M) { { _Val < _M }; };
	template <typename _Type>
	concept HasClampOp = HasClampMaxOp<_Type> && HasClampMinOp<_Type>;

	template <typename _Type, typename = std::enable_if_t<HasClampMaxOp<_Type>>>
	constexpr _Type ClampMax(const _Type& Value, const _Type& Max)
	{
		return Value > Max ? Max : Value;
	}

	template <typename _Type, typename = std::enable_if_t<HasClampMinOp<_Type>>>
	constexpr _Type ClampMin(const _Type& Value, const _Type& Min)
	{
		return Value < Min ? Min : Value;
	}

	template <typename _Type, typename = std::enable_if_t<HasClampOp<_Type>>>
	constexpr _Type Clamp(const _Type& Value, const _Type& Min, const _Type& Max)
	{
		return Value < Min ? Min : Value > Max ? Max : Value;
	}
}

namespace ComparisonOperators
{

}

_D_Dragonian_Lib_Operator_Space_End
