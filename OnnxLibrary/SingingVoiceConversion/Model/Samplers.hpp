/**
 * @file Diffusion-Svc.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Diffusion based Singing Voice Conversion
 * @changes
 *  > 2025/3/25 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Util/Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

using DiffusionSampler = Ort::Value(*)(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const DiffusionParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _DenoiseFn,
	const OnnxRuntimeModel& _NoisePredictorFn,
	const OnnxRuntimeModel& _AlphaCumprodFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& _Logger
	);

void RegisterDiffusionSampler(const std::wstring& _Name, DiffusionSampler _Sampler);
DiffusionSampler GetDiffusionSampler(const std::wstring& _Name);

using ReflowSampler = Ort::Value(*)(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& _Logger
	);

void RegisterReflowSampler(const std::wstring& _Name, ReflowSampler _Sampler);
ReflowSampler GetReflowSampler(const std::wstring& _Name);

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End