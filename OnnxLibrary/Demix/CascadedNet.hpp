/**
 * @file CascadedNet.hpp
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
 * @brief CascadedNet
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once

#include "OnnxLibrary/Demix/Base.hpp"
#include "Libraries/AvCodec/SignalProcess.h"

_D_Dragonian_Lib_Onnx_Demix_Header

class CascadedNet : public OnnxModelBase<CascadedNet>, public DemixModel
{
public:
	static HyperParameters GetPreDefinedHParams(
		const std::wstring& Name
	);

	CascadedNet() = delete;
	CascadedNet(
		const std::wstring& ModelPath,
		const OnnxRuntimeEnvironment& Environment,
		const HyperParameters& Setting = GetPreDefinedHParams(L"4band_v2"),
		const DLogger& Logger = _D_Dragonian_Lib_Onnx_Demix_Space GetDefaultLogger()
	);
	~CascadedNet() override = default;
	CascadedNet(const CascadedNet&) = default;
	CascadedNet(CascadedNet&&) noexcept = default;
	CascadedNet& operator=(const CascadedNet&) = default;
	CascadedNet& operator=(CascadedNet&&) noexcept = default;

	TemplateLibrary::Vector<SignalTensor> Forward(
		const SignalTensor& Signal,
		const Parameters& Params
	) const override;

private:
	std::tuple<SignalTensor, SpecTensor, SpecTensor, SpecTensor, Int64, Int64, Float32, Int64> Preprocess(
		const Tensor<Float32, 2, Device::CPU>& Signal,
		Int64 SamplingRate
	) const;

	SignalTensor Spec2Audio(
		const Tensor<Complex32, 3, Device::CPU>& Spec,
		const Tensor<Complex32, 3, Device::CPU>& InputHighEnd,
		Int64 InputHighEndH
	) const;

	CascadedNetConfig _MySetting;
	std::vector<FunctionTransform::StftKernel> _MyStftKernels;
	Int64 _MyPaddingLeft;
	Int64 _MyRoiSize;
	Signal::ResampleKernel<float> _MyResampleKernel{ FunctionTransform::KaiserWindow<float>(32) };
};

_D_Dragonian_Lib_Onnx_Demix_End