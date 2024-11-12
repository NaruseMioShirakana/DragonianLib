/**
 * FileName: Modules.hpp
 * Note: MoeVoiceStudioCore component management
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include "Models/VitsSvc.hpp"
#include "Models/DiffSvc.hpp"
#include "Models/ReflowSvc.hpp"
#include "Stft/stft.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @brief Gets the Mel operator for the given parameters
 * @param _SamplingRate Sampling rate
 * @param _Hopsize Hop size
 * @param _MelBins Number of Mel bins
 * @return Reference to the Mel operator
 */
FunctionTransform::Mel& GetMelOperator(
	int32_t _SamplingRate,
	int32_t _Hopsize,
	int32_t _MelBins
);

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End

namespace MoeVSRename
{
	// Namespace for MoeVSRename related functions and classes
}

