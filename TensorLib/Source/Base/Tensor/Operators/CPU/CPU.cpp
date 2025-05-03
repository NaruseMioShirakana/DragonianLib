#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

ThreadPool& GetThreadPool()
{
	static ThreadPool GlobalThreadPool{ 1 };
	return GlobalThreadPool;
}
ThreadPool& GetTaskPool()
{
	static ThreadPool GlobalTaskPool{ 1, L"[DragonianLib]", L"Worker of user defined operators" };
	return GlobalTaskPool;
}
void SetTaskPoolSize(SizeType _Size)
{
	GetTaskPool().Init(_Size);
}
SizeType& GetMaxTaskCountPerOperator()
{
	static SizeType GlobalMaxTaskCountPerOperator = 1;
	return GlobalMaxTaskCountPerOperator;
}
void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount)
{
	GetMaxTaskCountPerOperator() = _MaxTaskCount;
}
std::atomic_bool& GetInstantRunFlag()
{
	static std::atomic_bool InstantRunFlag = false;
	return InstantRunFlag;
}
void SetInstantRunFlag(bool _Flag)
{
	GetInstantRunFlag() = _Flag;
}
std::atomic_uint64_t& GetRandomDeviceId()
{
	static std::atomic_uint64_t RandomDeviceId = 0;
	return RandomDeviceId;
}

_D_Dragonian_Lib_Operator_Space_End