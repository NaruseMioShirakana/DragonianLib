/**
 * @file FireFlyGan.hpp
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
 * @brief Firefly Architecture
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once	
#include "OnnxLibrary/Vocoder/Vocoder.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

namespace FireflyArchitecture
{
	class Encoder : public OnnxModelBase<Encoder>
	{
	public:
		Encoder(
			const OnnxRuntimeEnvironment& _Environment,
			const std::wstring& _ModelPath,
			const std::shared_ptr<Logger>& _Logger = nullptr,
			int64_t SamplingRate = 44100,
			int64_t NumCodebooks = 8
		);
		~Encoder() = default;

		Encoder(const Encoder&) = default;
		Encoder& operator=(const Encoder&) = default;
		Encoder(Encoder&&) noexcept = default;
		Encoder& operator=(Encoder&&) noexcept = default;

		Tensor<Int64, 4, Device::CPU> Forward(
			const Tensor<Float32, 3, Device::CPU>& _Audio
		) const;

		int64_t GetSamplingRate() const
		{
			return _MySampleingRate;
		}

		int64_t GetNumCodebooks() const
		{
			return _MyNumCodebooks;
		}

	private:
		int64_t _MySampleingRate = 44100;
		int64_t _MyNumCodebooks = 8;
	};

	class Decoder : public OnnxModelBase<Decoder>
	{
	public:
		Decoder(
			const OnnxRuntimeEnvironment& _Environment,
			const std::wstring& _ModelPath,
			const std::shared_ptr<Logger>& _Logger = nullptr,
			int64_t SamplingRate = 44100,
			int64_t NumCodebooks = 8
		);
		~Decoder() = default;

		Decoder(const Decoder&) = default;
		Decoder& operator=(const Decoder&) = default;
		Decoder(Decoder&&) noexcept = default;
		Decoder& operator=(Decoder&&) noexcept = default;

		Tensor<Float32, 3, Device::CPU> Forward(
			const Tensor<Int64, 4, Device::CPU>& _Indices
		) const;

		int64_t GetSamplingRate() const
		{
			return _MySampleingRate;
		}

		int64_t GetNumCodebooks() const
		{
			return _MyNumCodebooks;
		}

	private:
		int64_t _MySampleingRate = 44100;
		int64_t _MyNumCodebooks = 8;
	};
}

_D_Dragonian_Lib_Onnx_Vocoder_End