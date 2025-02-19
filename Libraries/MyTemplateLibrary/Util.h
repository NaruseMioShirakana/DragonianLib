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

template <typename _Type>
class Ranges
{
public:
	_D_Dragonian_Lib_Constexpr_Force_Inline Ranges() = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline Ranges(_Type* _Begin, _Type* _End) : _MyBegin(_Begin), _MyEnd(_End) {}

	_D_Dragonian_Lib_Constexpr_Force_Inline _Type* Begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline _Type* End() const { return _MyEnd; }
	_D_Dragonian_Lib_Constexpr_Force_Inline _Type* begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline _Type* end() const { return _MyEnd; }

	_D_Dragonian_Lib_Constexpr_Force_Inline UInt64 Size() const { return _MyEnd - _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline _Type& operator[](size_t _Index) const { return _MyBegin[_Index]; }

	template <typename _Type2, typename=std::enable_if_t<std::is_assignable_v<_Type, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline Ranges& operator=(const Ranges<_Type2>& _Right)
	{
		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right[i];
		return *this;
	}
	
	template <typename _Type2, typename = std::enable_if_t<std::is_assignable_v<_Type, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline Ranges& operator=(const _Type2& _Right)
	{
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right;
		return *this;
	}

protected:
	_Type* _MyBegin = nullptr;
	_Type* _MyEnd = nullptr;
};

_D_Dragonian_Lib_Space_End




