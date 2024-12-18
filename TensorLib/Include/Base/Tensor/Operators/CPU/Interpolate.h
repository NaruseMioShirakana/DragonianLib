#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

/**
 * @brief Enum class representing interpolation mode.
 */
enum class InterpolateMode
{
	Nearest, ///< Nearest neighbor interpolation
	Linear, ///< Linear interpolation
	Bilinear, ///< Bilinear interpolation
	Bicubic, ///< Bicubic interpolation
	Trilinear, ///< Trilinear interpolation
	Area, ///< Area interpolation
};

template <size_t _Rank>
struct InterpolateParam
{
	InterpolateMode _MyMode = InterpolateMode::Nearest;
	std::optional<IDLArray<SizeType, _Rank>> _MySize;
	std::optional<IDLArray<float, _Rank>> _MyScale;

};

void InterpolateNearest(


);

_D_Dragonian_Lib_Operator_Space_End