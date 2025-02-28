#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

ThreadPool _Valdef_My_Thread_Pool{ 1 };
SizeType _Valdef_Global_Max_Task_Count_Per_Operator = 1;
bool _Flag_Instant_Run = true;
std::atomic_uint64_t _Valdef_Global_Random_Device_Id = 0;

ThreadPool& GetThreadPool()
{
	return _Valdef_My_Thread_Pool;
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