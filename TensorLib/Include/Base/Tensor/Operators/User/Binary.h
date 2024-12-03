#pragma once
#include "../OperatorBase.h"

#define DefDragonianLibBinaryOperator(_FunctionName, _FunctionType) \
	template <typename _Type> \
	std::enable_if_t<::DragonianLib::TypeTraits::IsSameTypeValue<_Type, _FunctionType##>, _Type> _FunctionName(const _Type& _Value1, const _Type& _Value2)

#define DefDragonianLibComparisonOperator(_FunctionName, _FunctionType) \
	template <typename _Type> \
	std::enable_if_t<::DragonianLib::TypeTraits::IsSameTypeValue<_Type, _FunctionType##>, bool> _FunctionName(const _Type& _Value1, const _Type& _Value2)

#define DefDragonianLibBinaryInplaceOperator(_FunctionName, _FunctionType) \
	template <typename _Type> \
	std::enable_if_t<::DragonianLib::TypeTraits::IsSameTypeValue<_Type, _FunctionType##>, _Type&> _FunctionName(_Type& _Value1, const _Type& _Value2)

_D_Dragonian_Lib_Operator_Space_Begin

namespace BinaryOperators
{
	
}

namespace ComparisonOperators
{

}

_D_Dragonian_Lib_Operator_Space_End
