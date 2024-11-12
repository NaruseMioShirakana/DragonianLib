#include "../BaseF0Extractor.hpp"
#include "Base.h"

_D_Dragonian_Lib_F0_Extractor_Header

Vector<float> BaseF0Extractor::ExtractF0(
	const Vector<Float64>& PCMData,
	const F0ExtractorParams& Params
)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

Vector<float> BaseF0Extractor::ExtractF0(
	const Vector<Float32>& PCMData,
	const F0ExtractorParams& Params
)
{
	return ExtractF0(
		SignalCast<Float64>(PCMData),
		Params
	);
}

Vector<float> BaseF0Extractor::ExtractF0(
	const Vector<Int16>& PCMData,
	const F0ExtractorParams& Params
)
{
	return ExtractF0(
		SignalCast<Float64>(PCMData),
		Params
	);
}

_D_Dragonian_Lib_F0_Extractor_End