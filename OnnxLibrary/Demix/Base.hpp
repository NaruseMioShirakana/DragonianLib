/**
 * @file Base.hpp
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
 * @brief Base header for Audio Demix
 * @changes
 *  > 2025/4/16 NaruseMioShirakana Created <
 */
#pragma once

#include "OnnxLibrary/Base/OrtBase.hpp"
#include "Libraries/Stft/Stft.hpp"

#define _D_Dragonian_Lib_Onnx_Demix_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace AudioDemix \
	{

#define _D_Dragonian_Lib_Onnx_Demix_End \
	} \
	_D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Onnx_Demix_Space \
	_D_Dragonian_Lib_Onnx_Runtime_Space \
	AudioDemix::

_D_Dragonian_Lib_Onnx_Demix_Header

DLogger& GetDefaultLogger() noexcept;

using SignalTensor = Tensor<Float32, 3, Device::CPU>;
using SpecTensor = Tensor<Complex32, 3, Device::CPU>;
using BatchedSpecTensor = Tensor<Complex32, 4, Device::CPU>;

struct Parameters
{
	Int64 SamplingRate;
	Int64 SplitBin;
	Float32 Value;
	Int64 SegmentSize = 512;
	ProgressCallback Progress = nullptr;
};

struct BandSetting
{
	Int64 SamplingRate = 0;
	Int64 HopSize = 0;
	Int64 FFTSize = 0;
	Int64 CropStart = 0;
	Int64 CropStop = 0;
	Int64 HpfStart = 0;
	Int64 HpfStop = 0;
	Int64 LpfStart = 0;
	Int64 LpfStop = 0;
};

struct DemixConfig
{
	Int64 SamplingRate = 44100;
	Int32 NumStft = 2048;
	Int32 HopSize = 512;
	Int32 WindowSize = 1024;

	Int64 SubBandCount = 1;
	Int64 StftBins = 1025;
	Boolean ComplexAsChannel = true;

	bool Center = true;
	PaddingType Padding = PaddingType::Reflect;
};

struct CascadedNetConfig
{
	Int64 Bins = 0;
	Int64 UnstableBins = 0;
	Int64 ReductionBins = 0;
	Int64 WindowSize = 0;
	std::vector<BandSetting> Bands;
	Int64 SamplingRate = 0;
	Int64 PreFilterStart = 0;
	Int64 PreFilterStop = 0;
	Int64 Offset = 0;
};

struct HyperParameters
{
	DemixConfig Demix;
	CascadedNetConfig Cascaded;
};

class DemixModel
{
public:
	DemixModel() = default;
	virtual ~DemixModel() = default;
	DemixModel(const DemixModel&) = default;
	DemixModel(DemixModel&&) noexcept = default;
	DemixModel& operator=(const DemixModel&) = default;
	DemixModel& operator=(DemixModel&&) noexcept = default;

	virtual TemplateLibrary::Vector<SignalTensor> Forward(
		const SignalTensor& Signal,
		const Parameters& Params
	) const = 0;
};

_D_Dragonian_Lib_Onnx_Demix_End