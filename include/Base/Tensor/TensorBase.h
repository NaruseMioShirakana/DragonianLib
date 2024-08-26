#pragma once
#include <complex>
#include <set>
#include <unordered_map>
#include "Value.h"

DragonianLibSpaceBegin
using SizeType = int64;
template <typename _Ty>
using Vector = std::vector<_Ty>;
template <typename _Ty>
using ContainerSet = std::set<_Ty>;
template <typename _TyA, typename _TyB>
using UnorderedMap = std::unordered_map<_TyA, _TyB>;
template <typename _TyA, typename _TyB>
using ContainerMap = std::unordered_map<_TyA, _TyB>;

enum class TensorType
{
	Boolean,
	Complex32,
	Float16,
	Float32,
	Float64,
	Int8,
	Int16,
	Int32,
	Int64,
	Float8,
	BFloat16,
	Complex64
};

int64 DType2Size(TensorType _Type);
int64 Type2Size(const std::string& _Type);
const std::string& DType2Type(TensorType _Type);
TensorType Type2DType(const std::string& _Type);

class TensorBase : public Value
{
public:
	TensorBase() = default;
	TensorBase(TensorType _DType = TensorType::Float32);
	TensorBase(const TensorBase& _Left) = delete;
	TensorBase& operator=(const TensorBase& _Left) = delete;
	TensorBase(TensorBase&& _Right) noexcept = delete;
	TensorBase& operator=(TensorBase&& _Right) noexcept = delete;
	~TensorBase() override = default;

protected:
	TensorType DType_ = TensorType::Float32;

public:
	TensorType DType() const;
};

DragonianLibSpaceEnd