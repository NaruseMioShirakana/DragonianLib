#include <random>
#include "Tensor/Tensor.h"

_D_Dragonian_Lib_Space_Begin

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
	Operators::_Valdef_My_Thread_Pool.SetRandomSeed(_Seed);
	Operators::_Valdef_Global_Random_Device_Id = 0;
}

void SetWorkerCount(SizeType _ThreadCount)
{
	Operators::_Valdef_My_Thread_Pool.Init(std::max(_ThreadCount, static_cast<SizeType>(0)));
	SetMaxTaskCountPerOperator(Operators::_Valdef_My_Thread_Pool.GetThreadCount() / 2);
}

void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount)
{
	Operators::_Valdef_Global_Max_Task_Count_Per_Operator = std::max(_MaxTaskCount, static_cast<SizeType>(1));
}

void EnableTimeLogger(bool _Enable)
{
	Operators::_Valdef_My_Thread_Pool.EnableTimeLogger(_Enable);
}

void EnableInstantRun(bool _Enable)
{
	Operators::_Valdef_My_Thread_Pool.EnableInstantRun(_Enable);
	Operators::_Flag_Instant_Run = _Enable;
}


_D_Dragonian_Lib_Space_End