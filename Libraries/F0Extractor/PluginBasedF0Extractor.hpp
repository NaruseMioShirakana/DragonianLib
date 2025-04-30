/**
 * @file PluginBasedF0Extractor.hpp
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
 * @brief F0 extractor based on dynamic library plugins
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/F0Extractor/BaseF0Extractor.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_F0_Extractor_Header

/**
 * @class PluginF0Extractor
 * @brief F0 extractor based on dynamic library plugins
 */
class PluginF0Extractor : public BaseF0Extractor
{
public:
	/**
	 * @brief GetF0Size function type (Instance, SamplingRate, HopSize, UserParameters)
	 */
	using GetF0SizeFunctionType = Int64(*)(
		void* Instance,
		long SamplingRate,
		long HopSize,
		void* UserParameters
		);

	/**
	 * @brief Extract function type (Instance, Input, InputSize, Channel, SamplingRate, HopSize, F0Bins, F0Max, F0Min, UserParameters, Output)
	 */
	using ExtractFunctionType = void(*)(
		void* Instance,
		const void* Input,
		Int64 InputSize, Int64 Channel,
		long SamplingRate, long HopSize, long F0Bins, double F0Max, double F0Min, void* UserParameters,
		float* Output
		);

	PluginF0Extractor(const Plugin::Plugin& Plugin, const void* UserParameter);
	~PluginF0Extractor() noexcept override = default;
	PluginF0Extractor(const PluginF0Extractor&) noexcept = default;
	PluginF0Extractor(PluginF0Extractor&&) noexcept = default;
	PluginF0Extractor& operator=(const PluginF0Extractor&) noexcept = default;
	PluginF0Extractor& operator=(PluginF0Extractor&&) noexcept = default;

	/**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
	 * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
	 */
	Tensor<Float32, 2, Device::CPU> ExtractF0(
		const Tensor<Float64, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
	) const override;

	/**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
	 * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
	 */
	Tensor<Float32, 2, Device::CPU> ExtractF0(
		const Tensor<Float32, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
	) const override;

	/**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
	 * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
	 */
	Tensor<Float32, 2, Device::CPU> ExtractF0(
		const Tensor<Int16, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
	) const override;

private:
	std::shared_ptr<void> _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;

	//"Int64 GetF0Size(void*, long, long, void*)"
	GetF0SizeFunctionType _MyGetF0Size = nullptr;

	//"void ExtractF0PD(void*, const void*, Int64, Int64, long, long, long, double, double, void*, float*)"
	ExtractFunctionType _MyExtractPD = nullptr;

	//"void ExtractF0PS(void*, const void*, Int64, Int64, long, long, long, double, double, void*, float*)"
	ExtractFunctionType _MyExtractPS = nullptr;

	//"void ExtractF0I16(void*, const void*, Int64, Int64, long, long, long, double, double, void*, float*)"
	ExtractFunctionType _MyExtractI16 = nullptr;
};

_D_Dragonian_Lib_F0_Extractor_End