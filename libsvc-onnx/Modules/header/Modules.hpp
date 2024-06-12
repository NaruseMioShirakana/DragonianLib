/**
 * FileName: Modules.hpp
 * Note: MoeVoiceStudioCore组件管理
 *
 * Copyright (C) 2022-2023 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of MoeVoiceStudioCore library.
 * MoeVoiceStudioCore library is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * MoeVoiceStudioCore library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
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

namespace libsvc
{
	class UnionSvcModel
	{
	public:
		UnionSvcModel() = delete;
		~UnionSvcModel();

		UnionSvcModel(const libsvc::Hparams& Config,
			const libsvc::LibSvcModule::ProgressCallback& Callback,
			int ProviderID, int NumThread, int DeviceID);

		[[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(const libsvc::SingleSlice& _Slice, const libsvc::InferenceParams& _Params, size_t& _Process) const;

		[[nodiscard]] DragonianLibSTL::Vector<int16_t> InferPCMData(const DragonianLibSTL::Vector<int16_t>& _PCMData, long _SrcSamplingRate, const libsvc::InferenceParams& _Params) const;

		[[nodiscard]] DragonianLibSTL::Vector<int16_t> ShallowDiffusionInference(
			DragonianLibSTL::Vector<float>& _16KAudioHubert,
			const libsvc::InferenceParams& _Params,
			std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
			const DragonianLibSTL::Vector<float>& _SrcF0,
			const DragonianLibSTL::Vector<float>& _SrcVolume,
			const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
			size_t& Process,
			int64_t SrcSize
		) const;

		libsvc::SingingVoiceConversion* GetPtr() const;

		[[nodiscard]] int64_t GetMaxStep() const;

		[[nodiscard]] bool OldVersion() const;

		[[nodiscard]] const std::wstring& GetDiffSvcVer() const;

		[[nodiscard]] int64_t GetMelBins() const;

		[[nodiscard]] int GetHopSize() const;

		[[nodiscard]] int64_t GetHiddenUnitKDims() const;

		[[nodiscard]] int64_t GetSpeakerCount() const;

		[[nodiscard]] bool CharaMixEnabled() const;

		[[nodiscard]] long GetSamplingRate() const;

		void NormMel(DragonianLibSTL::Vector<float>& MelSpec) const;

		[[nodiscard]] bool IsDiffusion() const;
	private:
		libsvc::DiffusionSvc* Diffusion_ = nullptr;
		libsvc::ReflowSvc* Reflow_ = nullptr;
	};

	void SetupKernel();

	DlCodecStft::Mel& GetMelOperator(
		int32_t _SamplingRate,
		int32_t _Hopsize,
		int32_t _MelBins
	);
}

namespace MoeVSRename
{
	
}

