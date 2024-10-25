#include <random>
#include "Tensor/Tensor.h"

DragonianLibSpaceBegin

SizeType VectorMul(const Dimensions& _Input)
{
	SizeType All = 1;
	for (const auto i : _Input)
		All *= i;
	return All;
}

SizeType VectorMul(const SliceOptions& _Input)
{
	SizeType All = 1;
	for (const auto& i : _Input)
		All *= (i.End - i.Begin);
	return All;
}

Dimensions GetBeginIndices(const SliceOptions& _Input)
{
	Dimensions Ret;
	Ret.Reserve(_Input.Size());
	for (const auto& i : _Input)
		Ret.EmplaceBack(i.Begin);
	return Ret;
}

bool RangeIsAllNone(const Vector<Range>& _Input)
{
	for (const auto& i : _Input)
		if (!i.IsNone)
			return false;
	return true;
}

void SetRandomSeed(SizeType _Seed)
{
	Operators::RandomEngine.seed(_Seed);
}

DragonianLibSpaceEnd