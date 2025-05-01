#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

static inline ThreadPool _Valdef_My_Thread_Pool{ 1 };
static inline ThreadPool _Valdef_My_Task_Pool{ 1, L"[DragonianLib]", L"Worker of user defined operators" };
static inline SizeType _Valdef_Global_Max_Task_Count_Per_Operator = 1;
static inline bool _Flag_Instant_Run = true;
static inline std::atomic_uint64_t _Valdef_Global_Random_Device_Id = 0;

ThreadPool& GetThreadPool()
{
	return _Valdef_My_Thread_Pool;
}
ThreadPool& GetTaskPool()
{
	return _Valdef_My_Task_Pool;
}
void SetTaskPoolSize(SizeType _Size)
{
	_Valdef_My_Task_Pool.Init(_Size);
}
SizeType GetMaxTaskCountPerOperator()
{
	return _Valdef_Global_Max_Task_Count_Per_Operator;
}
void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount)
{
	_Valdef_Global_Max_Task_Count_Per_Operator = _MaxTaskCount;
}
bool GetInstantRunFlag()
{
	return _Flag_Instant_Run;
}
void SetInstantRunFlag(bool _Flag)
{
	_Flag_Instant_Run = _Flag;
}
std::atomic_uint64_t& GetRandomDeviceId()
{
	return _Valdef_Global_Random_Device_Id;
}

_D_Dragonian_Lib_Operator_Space_End