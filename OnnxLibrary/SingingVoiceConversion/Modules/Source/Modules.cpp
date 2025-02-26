#include "Libraries/Base.h"
#include "../header/Modules.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

std::unordered_map<std::wstring, FunctionTransform::Mel*> MelOperators;

FunctionTransform::Mel& GetMelOperator(
	int32_t _SamplingRate,
	int32_t _Hopsize,
	int32_t _MelBins
)
{
	const std::wstring _Name = L"S" +
		std::to_wstring(_SamplingRate) +
		L"H" + std::to_wstring(_Hopsize) +
		L"M" + std::to_wstring(_MelBins);
	if (!MelOperators.contains(_Name))
	{
		if (MelOperators.size() > 10)
		{
			delete MelOperators.begin()->second;
			MelOperators.erase(MelOperators.begin());
		}
		MelOperators[_Name] = new FunctionTransform::Mel(_Hopsize * 4, _Hopsize, _SamplingRate, _MelBins);
	}
	return *MelOperators[_Name];
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
