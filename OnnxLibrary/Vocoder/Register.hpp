/**
 * @file Register.hpp
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
 * @brief Register of Vocoder
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/Vocoder/Vocoder.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

using Vocoder = std::shared_ptr<VocoderBase>;
using Constructor = std::function<Vocoder(
	const std::wstring& _Path,
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate,
	Int64 _MelBins,
	const std::shared_ptr<Logger>& _Logger
)>;

/**
 * @brief Register a Vocoder
 * @param _PluginName Name of the Vocoder
 * @param _Constructor Constructor of the Vocoder
 */
void RegisterVocoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
);

/**
 * @brief Create a new Vocoder instance
 * @param Name Name of the Vocoder
 * @param _Path Path of the model
 * @param _Environment OnnxRuntime enviroment
 * @param _SamplingRate Sampling rate of the model
 * @param _MelBins MelBins of the model
 * @param _Logger Logger
 * @return New Vocoder instance
 */
Vocoder New(
	const std::wstring& Name,
	const std::wstring& _Path,
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate = 44100,
	Int64 _MelBins = 128,
	const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Vocoder_Space GetDefaultLogger()
);

/**
 * @brief Get a list of Vocoder names
 * @return List of Vocoder names
 */
const std::vector<std::wstring>& GetVocoderList();

_D_Dragonian_Lib_Onnx_Vocoder_End