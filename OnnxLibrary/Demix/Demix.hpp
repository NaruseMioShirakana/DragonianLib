/**
 * @file Demix.hpp
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
 * @brief Demix Models
 * @changes
 *  > 2025/4/16 NaruseMioShirakana Created <
 */
#pragma once
#include "OnnxLibrary/Demix/Base.hpp"

_D_Dragonian_Lib_Onnx_Demix_Header

class Demix : public OnnxModelBase<Demix>, public DemixModel
{
public:
	Demix() = delete;
	Demix(
		const std::wstring& ModelPath,
		const OnnxRuntimeEnvironment& Environment,
		const HyperParameters& HParams,
		const DLogger& Logger = _D_Dragonian_Lib_Onnx_Demix_Space GetDefaultLogger()
	);
	~Demix() override = default;
	Demix(const Demix&) = default;
	Demix(Demix&&) noexcept = default;
	Demix& operator=(const Demix&) = default;
	Demix& operator=(Demix&&) noexcept = default;

	TemplateLibrary::Vector<SignalTensor> Forward(
		const SignalTensor& Signal,
		const Parameters& Params
	) const override;

	Tensor<Complex32, 5, Device::CPU> ExecuteModel(
		const Tensor<Float32, 5, Device::CPU>& Signal,
		const Parameters& Params
	) const;

private:
	Int64 _MySamplingRate = 44100;
	Int64 _MySubBandCount = 1;
	Int64 _MyStftBins = 1025;
	bool _ComplexAsChannel = true;
	FunctionTransform::StftKernel _MyStftKernel;
};

_D_Dragonian_Lib_Onnx_Demix_End