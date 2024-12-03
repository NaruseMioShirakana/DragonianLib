#pragma once
#include "../OperatorBase.h"

#define DefDragonianLibUnaryOperator(_FunctionName, _FunctionType) \
	template <typename _Type> \
	std::enable_if_t<::DragonianLib::TypeTraits::IsSameTypeValue<_Type, _FunctionType##>, _Type> _FunctionName(const _Type& _Value)

_D_Dragonian_Lib_Operator_Space_Begin

namespace UnaryOperators
{
	
}

_D_Dragonian_Lib_Operator_Space_End