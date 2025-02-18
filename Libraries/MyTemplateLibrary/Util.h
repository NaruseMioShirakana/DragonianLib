#pragma once

#include "../Base.h"

#define _D_Dragonian_Lib_Template_Library_Space_Begin _D_Dragonian_Lib_Space_Begin namespace TemplateLibrary {
#define _D_Dragonian_Lib_Template_Library_Space_End } _D_Dragonian_Lib_Space_End
#define _D_Dragonian_Lib_TL_Namespace _D_Dragonian_Lib_Namespace TemplateLibrary::

_D_Dragonian_Lib_Template_Library_Space_Begin

template <typename _Type>
class _MyLess
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A < _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type & _Left, const _Type & _Right)
	{
		return _Left < _Right;
	}

};

template <typename _Type>
class _MyGreater
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A > _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type& _Left, const _Type& _Right)
	{
		return _Left > _Right;
	}
};

template <typename _Type>
class _MyEqual
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A == _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (std::is_floating_point_v<_Type>)
			return std::abs(_Left - _Right) < std::numeric_limits<_Type>::epsilon();
		else
			return _Left == _Right;
	}
};

_D_Dragonian_Lib_Template_Library_Space_End

_D_Dragonian_Lib_Space_Begin

enum class Device
{
	CPU = 0,
	CUDA,
	HIP,
	DIRECTX,
	CUSTOM
};

static inline size_t NopID = size_t(-1);

namespace DragonianLibSTL
{
	using namespace _D_Dragonian_Lib_Namespace TemplateLibrary;
}

_D_Dragonian_Lib_Space_End




