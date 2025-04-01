/**
 * @file UVR.hpp
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
 * @brief Ultimate Vocal Remover
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once

#include "OnnxLibrary/Base/OrtBase.hpp"
#include "Libraries/Stft/Stft.hpp"

#define _D_Dragonian_Lib_Onnx_UVR_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace UltimateVocalRemover \
	{

#define _D_Dragonian_Lib_Onnx_UVR_End \
	} \
	_D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Onnx_UVR_Space \
	_D_Dragonian_Lib_Onnx_Runtime_Space \
	UltimateVocalRemover::

_D_Dragonian_Lib_Onnx_UVR_Header

DLogger& GetDefaultLogger() noexcept;

class CascadedNet : public OnnxModelBase<CascadedNet>
{
public:
	struct BandSetting
	{
		Int64 SamplingRate;
		Int64 HopSize;
		Int64 FFTSize;
		Int64 CropStart;
		Int64 CropStop;
		Int64 HpfStart;
		Int64 HpfStop;
		Int64 LpfStart;
		Int64 LpfStop;
	};
	struct HParams
	{
		Int64 Bins;
		Int64 UnstableBins;
		Int64 ReductionBins;
		Int64 WindowSize;
		std::vector<BandSetting> Bands;
		Int64 SamplingRate;
		Int64 PreFilterStart;
		Int64 PreFilterStop;
		Int64 Offset;
	};
	static HParams GetPreDefinedHParams(
		const std::wstring& Name
	);

	CascadedNet() = delete;
	CascadedNet(
		const std::wstring& ModelPath,
		const OnnxRuntimeEnvironment& Environment,
		HParams Setting = GetPreDefinedHParams(L"4band_v2"),
		const DLogger& Logger = _D_Dragonian_Lib_Onnx_UVR_Space GetDefaultLogger()
	);
	~CascadedNet() = default;
	CascadedNet(const CascadedNet&) = default;
	CascadedNet(CascadedNet&&) noexcept = default;
	CascadedNet& operator=(const CascadedNet&) = default;
	CascadedNet& operator=(CascadedNet&&) noexcept = default;

	std::tuple<Tensor<Float32, 3, Device::CPU>,
		Tensor<Complex32, 3, Device::CPU>,
		Tensor<Complex32, 3, Device::CPU>,
		Tensor<Complex32, 3, Device::CPU>,
		Int64, Int64, Float32, Int64> Preprocess(
			const Tensor<Float32, 2, Device::CPU>& Signal,
			Int64 SamplingRate
		) const;

	Tensor<Float32, 3, Device::CPU> Spec2Audio(
		const Tensor<Complex32, 3, Device::CPU>& Spec,
		const Tensor<Complex32, 3, Device::CPU>& InputHighEnd,
		Int64 InputHighEndH
	) const;

	Tensor<Float32, 3, Device::CPU> Forward(
		const Tensor<Float32, 2, Device::CPU>& Signal,
		Int64 SplitBin,
		Float32 Value,
		Int64 SamplingRate
	) const;

private:
	HParams _MySetting;
	std::vector<FunctionTransform::StftKernel> _MyStftKernels;
	Int64 _MyPaddingLeft;
	Int64 _MyRoiSize;
};

_D_Dragonian_Lib_Onnx_UVR_End