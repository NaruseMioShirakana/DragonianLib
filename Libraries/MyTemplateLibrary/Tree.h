#pragma once
#include "Alloc.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

template <typename _ValueType, UInt64 _Rank, bool _Plus, typename _Alloc = CPUAllocator>
struct NRankTreeNode
{
	static_assert(_Rank > 1, "The rank of the tree node must be greater than 1.");
	using ValueType = _ValueType;
	using MyNodePointer = NRankTreeNode*;

	static constexpr UInt64 _MyRank = _Rank;
	static constexpr UInt64 _MyMaxChildCount = _MyRank;
	static constexpr UInt64 _MyMaxNodeCount = _Plus ? _MyRank : _MyRank - 1;
	static constexpr UInt64 _MyMinChildCount = _MyRank % 2 ? _MyRank / 2 + 1 : _MyRank / 2;
	static constexpr UInt64 _MyMinNodeCount = _Plus ? _MyMinChildCount : _MyMinChildCount - 1;

	_D_Dragonian_Lib_Constexpr_Force_Inline static Int64 GetHeight(const NRankTreeNode* _Node)
	{
		return _Node ? _Node->_MyHeight : 0;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Int64 GetHeight() const
	{
		return GetHeight(this);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, Int64> GetBalanceFactor(const NRankTreeNode* _Node)
	{
		return GetHeight(_Node->_MyChild[0]) - GetHeight(_Node->_MyChild[1]);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<_MyMaxChildCount == 2, Int64> GetBalanceFactor() const
	{
		return GetBalanceFactor(this);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, bool> IsRedNode(const NRankTreeNode* _Node)
	{
		return _Node ? _Node->_IsRed : false;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, MyNodePointer> LeftRotate(MyNodePointer _OldRoot)
	{
		auto _NewRoot = _OldRoot->_MyChild[1];
		auto _Parent = _OldRoot->_MyParent;
		_OldRoot->_MyChild[1] = _NewRoot->_MyChild[0];
		_NewRoot->_MyChild[0] = _OldRoot;
		_NewRoot->_MyParent = _Parent;
		_OldRoot->_MyParent = _NewRoot;
		if (_OldRoot->_MyChild[1])
			_OldRoot->_MyChild[1]->_MyParent = _OldRoot;
		if (_Parent)
		{
			if (_Parent->_MyChild[0] == _OldRoot)
				_Parent->_MyChild[0] = _NewRoot;
			else
				_Parent->_MyChild[1] = _NewRoot;
		}
		_OldRoot->_MyHeight = std::max(GetHeight(_OldRoot->_MyChild[0]), GetHeight(_OldRoot->_MyChild[1])) + 1;
		_NewRoot->_MyHeight = std::max(GetHeight(_NewRoot->_MyChild[0]), GetHeight(_NewRoot->_MyChild[1])) + 1;
		return _NewRoot;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, MyNodePointer> RightRotate(MyNodePointer _OldRoot)
	{
		auto _NewRoot = _OldRoot->_MyChild[0];
		auto _Parent = _OldRoot->_MyParent;
		_OldRoot->_MyChild[0] = _NewRoot->_MyChild[1];
		_NewRoot->_MyChild[1] = _OldRoot;
		_NewRoot->_MyParent = _Parent;
		_OldRoot->_MyParent = _NewRoot;
		if (_OldRoot->_MyChild[0])
			_OldRoot->_MyChild[0]->_MyParent = _OldRoot;
		if (_Parent)
		{
			if (_Parent->_MyChild[0] == _OldRoot)
				_Parent->_MyChild[0] = _NewRoot;
			else
				_Parent->_MyChild[1] = _NewRoot;
		}
		_OldRoot->_MyHeight = std::max(GetHeight(_OldRoot->_MyChild[0]), GetHeight(_OldRoot->_MyChild[1])) + 1;
		_NewRoot->_MyHeight = std::max(GetHeight(_NewRoot->_MyChild[0]), GetHeight(_NewRoot->_MyChild[1])) + 1;
		return _NewRoot;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, MyNodePointer> LeftRightRotate(MyNodePointer _GrandFather)
	{
		_GrandFather->_MyChild[0] = LeftRotate(_GrandFather->_MyChild[0]);
		return RightRotate(_GrandFather);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::enable_if_t<_MyMaxChildCount == 2, MyNodePointer> RightLeftRotate(MyNodePointer _GrandFather)
	{
		_GrandFather->_MyChild[1] = RightRotate(_GrandFather->_MyChild[1]);
		return LeftRotate(_GrandFather);
	}

	static _D_Dragonian_Lib_Constexpr_Force_Inline void _MyDeleter(void* _Block)
	{
		if (_Block)
		{
			if constexpr (!std::is_trivially_copy_assignable_v<ValueType>)
			{
				ValueType* _Value = static_cast<ValueType*>(_Block);
				for (UInt64 i = 0; i < _MyMaxNodeCount; ++i)
					_Value[i].~ValueType();
			}
			_Alloc::deallocate(_Block);
		}
	}
	template <typename = std::enable_if_t<!std::is_constructible_v<ValueType>>>
	NRankTreeNode() = delete;
	template <typename... _ArgTys, typename = std::enable_if_t<std::is_constructible_v<ValueType, _ArgTys...>>>
	NRankTreeNode(_ArgTys&&... _Args) :
		_MyBuffer(_Alloc::allocate(sizeof(ValueType)* _MyMaxNodeCount), _MyDeleter),
		_MyValue(static_cast<ValueType*>(_MyBuffer.get()))
	{
		new (_MyValue) ValueType(std::forward<_ArgTys>(_Args)...);
		if constexpr (_MyMaxNodeCount > 1)
			for (UInt64 i = 1; i < _MyMaxNodeCount; ++i)
				new (_MyValue + i) ValueType(*_MyValue);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const
	{
		return &_MyValue[0];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const
	{
		return &_MyValue[_MyMaxNodeCount - 1];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](UInt64 _Index) const
	{
		return _MyValue[_Index];
	}

	template <typename... _ArgTys, typename = std::enable_if_t<std::is_constructible_v<ValueType, _ArgTys...>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Emplace(UInt64 _Index, _ArgTys&&... _Args)
	{
		_MyValue[_Index] = ValueType(std::forward<_ArgTys>(_Args)...);
		return _MyValue[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) At(UInt64 _Index) const
	{
		return _MyValue[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Front() const
	{
		return _MyValue[0];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Back() const
	{
		return _MyValue[_MyMaxNodeCount - 1];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetParent() const
	{
		return _MyParent;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetLeftBrother() const
	{
		if constexpr (_Plus)
			return _MyBrother[0]._MyValue;
		else
			return _MyBrother[0];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetRightBrother() const
	{
		if constexpr (_Plus)
			return _MyBrother[1]._MyValue;
		else
			return _MyBrother[1];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetChild(UInt64 _Index) const
	{
		return _MyChild[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetLeftChild() const
	{
		return _MyChild[0];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetRightChild() const
	{
		return _MyChild[1];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetLeftChild(UInt64 _Index) const
	{
		return _MyChild[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetRightChild(UInt64 _Index) const
	{
		return _MyChild[_Index + 1];
	}

	template <typename... _ArgTys, typename = std::enable_if_t<std::is_constructible_v<ValueType, _ArgTys...>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline static decltype(auto) Insert(MyNodePointer& _Root, _ArgTys&&... _Args)
	{
		if (!_Root)
			return _Root = _CreateNode(std::forward<_ArgTys>(_Args)...);
		auto _Current = _Root;
		auto _Parent = _Root;
		auto _Item = _CreateNode(std::forward<_ArgTys>(_Args)...);
		while (_Current)
		{
			_Parent = _Current;
			if (_MyPred(_Item->_MyValue, _Current->_MyValue))
				_Current = _Current->_MyChild[0];
			else
				_Current = _Current->_MyChild[1];
		}

		return _Root;
	}

protected:
	std::shared_ptr<void> _MyBuffer = nullptr;
	ValueType* _MyValue;
	MyNodePointer _MyParent = nullptr;
	MyNodePointer _MyChild[_MyMaxChildCount]{ nullptr };
	Int64 _MyHeight = 1;
	OptionalType<MyNodePointer, _Plus> _MyBrother[2]{ 0, 0 };
	bool _IsRed = true;
};

template <typename _ValueType, typename _Pred = std::less<_ValueType>, typename _Alloc = CPUAllocator>
class BinaryTree
{
public:
	using _MyNodeType = NRankTreeNode<_ValueType, 2, false, _Alloc>;
	using _MyValueType = typename _MyNodeType::ValueType;
	using _MySizeType = UInt64;
	using _MyNodePointer = _MyNodeType*;
	using _MyNodeReference = _MyNodeType&;
	using _MyValuePointer = _MyValueType*;
	using _MyValueReference = _MyValueType&;
	using _MyValueConstReference = const _MyValueType&;
	using _KeyPred = _Pred;

protected:
	_MyNodePointer* _MyRoot = nullptr;
	_MySizeType _MySize = 0;
	_KeyPred _MyPred;

	template <typename... _ArgTys, typename = std::enable_if_t<std::is_constructible_v<_MyValueType, _ArgTys...>>>
	static decltype(auto) _CreateNode(_ArgTys&&... _Args)
	{
		auto _Node = (_MyNodePointer)_Alloc::allocate(sizeof(_MyNodeType));
		new (_Node) _MyNodeType(std::forward<_ArgTys>(_Args)...);
		return _Node;
	}
	static void _DestroyNode(_MyNodePointer _Node)
	{
		if (!_Node)
			return;
		_Node->~_MyNodeType();
		_Alloc::deallocate(_Node);
	}
};

template <typename _ValueType, typename _Pred = std::less<_ValueType>, typename _Alloc = CPUAllocator>
class AVLTree : public BinaryTree<_ValueType, _Pred, _Alloc>
{
public:
	using _MyBase = BinaryTree<_ValueType, _Pred, _Alloc>;
	using _MyNodeType = typename _MyBase::_MyNodeType;
	using _MyValueType = typename _MyBase::_MyValueType;
	using _MySizeType = typename _MyBase::_MySizeType;
	using _MyNodePointer = typename _MyBase::_MyNodePointer;
	using _MyNodeReference = typename _MyBase::_MyNodeReference;
	using _MyValuePointer = typename _MyBase::_MyValuePointer;
	using _MyValueReference = typename _MyBase::_MyValueReference;
	using _MyValueConstReference = typename _MyBase::_MyValueConstReference;

private:
	using _MyBase::_CreateNode;
	using _MyBase::_DestroyNode;

protected:
	using _MyBase::_MyRoot;
	using _MyBase::_MySize;
	using _MyBase::_MyPred;

public:
	template <typename... _ArgTys, typename = std::enable_if_t<std::is_constructible_v<_MyValueType, _ArgTys...>>>
	decltype(auto) Insert(_ArgTys&&... _Args)
	{
		auto _Node = _CreateNode(std::forward<_ArgTys>(_Args)...);
		if (!_MyRoot)
		{
			_MyRoot = _Node;
			_MySize = 1;
			return _Node;
		}
		auto _Current = _MyRoot;
		while (true)
		{
			if (_MyPred(_Node->_MyValue, _Current->_MyValue))
			{
				if (!_Current->_MyChild[0])
				{
					_Current->_MyChild[0] = _Node;
					_Node->_MyParent = _Current;
					break;
				}
				_Current = _Current->_MyChild[0];
			}
			else
			{
				if (!_Current->_MyChild[1])
				{
					_Current->_MyChild[1] = _Node;
					_Node->_MyParent = _Current;
					break;
				}
				_Current = _Current->_MyChild[1];
			}
		}
	}
};




_D_Dragonian_Lib_Template_Library_Space_End