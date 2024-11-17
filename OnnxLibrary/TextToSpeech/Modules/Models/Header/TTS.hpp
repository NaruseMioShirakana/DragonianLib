/**
 * FileName: TTS.hpp
 * Note: MoeVoiceStudioCore TTS基类
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
 * date: 2023-11-9 Create
*/

#pragma once
#include <map>
#include "Params.hpp"
#include "Dict/Dict.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

class TextToSpeech : public LibTTSModule
{
public:
	using DurationCallback = std::function<void(float*, const float*)>;

	TextToSpeech(
		const ModelHParams& HParams,
		const ExecutionProviders& ExecutionProvider_,
		unsigned DeviceID_,
		unsigned ThreadCount_ = 0
	);

	DragonianLibSTL::Vector<float> Inference(
		TTSInputData& InputData,
		TTSParams& Params,
		bool Inference
	) const;

	virtual DragonianLibSTL::Vector<float> Inference(
		TTSInputData& InputData,
		const TTSParams& Params
	) const;

	[[nodiscard]] DragonianLibSTL::Vector<int64_t> CleanedSeq2Indices(
		const DragonianLibSTL::Vector<std::wstring>& Seq
	) const;

	[[nodiscard]] DragonianLibSTL::Vector<int64_t> LanguageSymbol2Indices(
		const DragonianLibSTL::Vector<std::wstring>& Seq,
		int64_t LanguageID
	) const;

	[[nodiscard]] std::map<int64_t, float> SpeakerMixSymbol2Indices(
		const DragonianLibSTL::Vector<std::pair<std::wstring, float>>& Seq,
		int64_t SpeakerID
	) const;

	~TextToSpeech() override = default;

protected:
	std::unordered_map<std::wstring, int64_t> SpeakerName2ID;
	std::unordered_map<int64_t, std::wstring> SpeakerID2Name;
	std::unordered_map<std::wstring, int64_t> Symbols;
	std::unordered_map<std::wstring, int64_t> Language2ID;

public:
	[[nodiscard]] static DragonianLibSTL::Vector<DragonianLibSTL::Vector<bool>> generatePath(
		float* duration,
		size_t durationSize,
		size_t maskSize
	);
	[[nodiscard]] static DragonianLibSTL::Vector<int64_t> GetAligments(
		size_t DstLen,
		size_t SrcLen
	);

protected:
	bool AddBlank = true;
	int64_t SpeakerCount = 1;
	int64_t UNKID = 0;
	int64_t PADID = 0;
	DurationCallback CustomDurationCallback;

public:
	TextToSpeech(const TextToSpeech&) = default;
	TextToSpeech& operator=(const TextToSpeech&) = default;
	TextToSpeech(TextToSpeech&&) noexcept = default;
	TextToSpeech& operator=(TextToSpeech&&) noexcept = default;
};

class ContextModel : public LibTTSModule
{
public:
	ContextModel(
		const std::wstring& ModelPath,
		const ExecutionProviders& ExecutionProvider_,
		unsigned DeviceID_,
		unsigned ThreadCount_ = 0
	);

	~ContextModel() override = default;

	std::pair<DragonianLibSTL::Vector<float>, int64_t> Inference(
		const std::wstring& InputData,
		Dict::Tokenizer::TokenizerMethod _Method = Dict::Tokenizer::Maximum,
		bool _SkipNonLatin = true,
		size_t _MaximumMatching = 12
	) const;

	std::pair<DragonianLibSTL::Vector<float>, int64_t> Inference(
		DragonianLibSTL::Vector<int64_t>& TokenIds
	) const;

	ContextModel(const ContextModel&) = default;
	ContextModel& operator=(const ContextModel&) = default;
	ContextModel(ContextModel&&) noexcept = default;
	ContextModel& operator=(ContextModel&&) noexcept = default;

private:
	std::shared_ptr<Ort::Session> Session = nullptr;
	Dict::Tokenizer Tokenizer;
	static inline const std::vector<const char*> InputNames = { "input_ids", "attention_mask", "token_type_ids" };
	static inline const std::vector<const char*> OutputNames = { "last_hidden_state" };
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End