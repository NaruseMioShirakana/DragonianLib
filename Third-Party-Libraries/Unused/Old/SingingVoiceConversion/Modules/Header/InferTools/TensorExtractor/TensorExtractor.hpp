/**
 * FileName: MoeVSCoreTensorExtractor.hpp
 * Note: MoeVoiceStudioCore TensorExtractors
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
#include "BaseTensorExtractor.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_Header

class SoVits2TensorExtractor : public LibSvcTensorExtractor
{
public:
	SoVits2TensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~SoVits2TensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	SoVits2TensorExtractor(const SoVits2TensorExtractor&) = delete;
	SoVits2TensorExtractor(SoVits2TensorExtractor&&) = delete;
	SoVits2TensorExtractor operator=(const SoVits2TensorExtractor&) = delete;
	SoVits2TensorExtractor operator=(SoVits2TensorExtractor&&) = delete;

	const std::vector<const char*> InputNames = { "hidden_unit", "lengths", "pitch", "sid" };
};

class SoVits3TensorExtractor : public LibSvcTensorExtractor
{
public:
	SoVits3TensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~SoVits3TensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	SoVits3TensorExtractor(const SoVits3TensorExtractor&) = delete;
	SoVits3TensorExtractor(SoVits3TensorExtractor&&) = delete;
	SoVits3TensorExtractor operator=(const SoVits3TensorExtractor&) = delete;
	SoVits3TensorExtractor operator=(SoVits3TensorExtractor&&) = delete;

	const std::vector<const char*> InputNames = { "hidden_unit", "lengths", "pitch", "sid" };
};

class SoVits4TensorExtractor : public LibSvcTensorExtractor
{
public:
	SoVits4TensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~SoVits4TensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	SoVits4TensorExtractor(const SoVits4TensorExtractor&) = delete;
	SoVits4TensorExtractor(SoVits4TensorExtractor&&) = delete;
	SoVits4TensorExtractor operator=(const SoVits4TensorExtractor&) = delete;
	SoVits4TensorExtractor operator=(SoVits4TensorExtractor&&) = delete;

	const std::vector<const char*> InputNames = { "c", "f0", "mel2ph", "uv", "noise", "sid" };
	const std::vector<const char*> InputNamesVol = { "c", "f0", "mel2ph", "uv", "noise", "sid", "vol" };
};

class SoVits4DDSPTensorExtractor : public LibSvcTensorExtractor
{
public:
	SoVits4DDSPTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~SoVits4DDSPTensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	SoVits4DDSPTensorExtractor(const SoVits4DDSPTensorExtractor&) = delete;
	SoVits4DDSPTensorExtractor(SoVits4DDSPTensorExtractor&&) = delete;
	SoVits4DDSPTensorExtractor operator=(const SoVits4DDSPTensorExtractor&) = delete;
	SoVits4DDSPTensorExtractor operator=(SoVits4DDSPTensorExtractor&&) = delete;

	const std::vector<const char*> InputNames = { "c", "f0", "mel2ph", "t_window", "noise", "sid" };
	const std::vector<const char*> InputNamesVol = { "c", "f0", "mel2ph", "t_window", "noise", "sid", "vol" };
};

class RVCTensorExtractor : public LibSvcTensorExtractor
{
public:
	RVCTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~RVCTensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	RVCTensorExtractor(const RVCTensorExtractor&) = delete;
	RVCTensorExtractor(RVCTensorExtractor&&) = delete;
	RVCTensorExtractor operator=(const RVCTensorExtractor&) = delete;
	RVCTensorExtractor operator=(RVCTensorExtractor&&) = delete;

	const std::vector<const char*> InputNames = { "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd" };
	const std::vector<const char*> InputNamesVol = { "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd", "vol" };
};

class DiffSvcTensorExtractor : public LibSvcTensorExtractor
{
public:
	DiffSvcTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~DiffSvcTensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	DiffSvcTensorExtractor(const DiffSvcTensorExtractor&) = delete;
	DiffSvcTensorExtractor(DiffSvcTensorExtractor&&) = delete;
	DiffSvcTensorExtractor operator=(const DiffSvcTensorExtractor&) = delete;
	DiffSvcTensorExtractor operator=(DiffSvcTensorExtractor&&) = delete;


	const std::vector<const char*> InputNames = { "hubert", "mel2ph", "spk_embed", "f0" };
	const std::vector<const char*> OutputNames = { "mel_pred", "f0_pred" };
};

class DiffusionSvcTensorExtractor : public LibSvcTensorExtractor
{
public:
	DiffusionSvcTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other) : LibSvcTensorExtractor(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other) {}
	~DiffusionSvcTensorExtractor() override = default;
	Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	) override;
	DiffusionSvcTensorExtractor(const DiffusionSvcTensorExtractor&) = delete;
	DiffusionSvcTensorExtractor(DiffusionSvcTensorExtractor&&) = delete;
	DiffusionSvcTensorExtractor operator=(const DiffusionSvcTensorExtractor&) = delete;
	DiffusionSvcTensorExtractor operator=(DiffusionSvcTensorExtractor&&) = delete;


	const std::vector<const char*> InputNamesVol = { "hubert", "mel2ph", "f0", "volume", "spk_mix", "randn" };
	const std::vector<const char*> InputNames = { "hubert", "mel2ph", "f0", "spk_mix", "randn" };
	const std::vector<const char*> OutputNames = { "mel_pred", "f0_pred", "init_noise" };
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_End