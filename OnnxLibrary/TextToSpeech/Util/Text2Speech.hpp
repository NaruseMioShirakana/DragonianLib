/**
 * @file Text2Speech.hpp
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
 * @brief Base TextToSpeech Module
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once
#include "Util.hpp"
#include "Libraries/Dict/Dict.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

DLogger& GetDefaultLogger() noexcept;

class EmotionEmbedding
{
public:
	EmotionEmbedding() = default;
	EmotionEmbedding(const std::wstring& EmotionFilePath, Int64 EmotionDims = 1024)
	{
		if (EmotionFilePath.empty())
			return;
		_MyEmotionDims = EmotionDims;
		_MyEmotionVectors = Functional::NumpyLoad<Float32, 2>(EmotionFilePath);
		if (_MyEmotionVectors.Shape(1) != _MyEmotionDims)
			_D_Dragonian_Lib_Throw_Exception("Emotion Vector Dims Not Match!");
	}
	Tensor<Float, 1, Device::CPU> operator[](Int64 Index) const
	{
		return _MyEmotionVectors[Index].Clone().Evaluate();
	}
private:
	Tensor<Float, 2, Device::CPU> _MyEmotionVectors;
	Int64 _MyEmotionDims = 1024;
};

/**
 * @class ContextModel
 * @brief Bert and Clap Model
 */
class ContextModel : public OnnxModelBase<ContextModel>
{
public:
	ContextModel() = delete;
	ContextModel(
		const OnnxRuntimeEnvironment& _Environment,
		const std::wstring& _Path,
		const std::shared_ptr<Logger>& _Logger = GetDefaultLogger()
	);
	~ContextModel() = default;
	ContextModel(const ContextModel&) = default;
	ContextModel& operator=(const ContextModel&) = default;
	ContextModel(ContextModel&&) noexcept = default;
	ContextModel& operator=(ContextModel&&) noexcept = default;

	//See Dict::Tokenizer
	Tensor<Float32, 3, Device::CPU> Forward(
		const Tensor<Int64, 2, Device::CPU>& TokenIds,
		const Tensor<Int64, 2, Device::CPU>& TokenTypeIds,
		std::optional<Tensor<Int64, 2, Device::CPU>> AttentionMask = std::nullopt
	) const;
};

Tensor<Int64, 1, Device::CPU> CleanedText2Indices(
	const std::wstring& Text,
	const std::unordered_map<std::wstring, Int64>& Symbols
);

Tensor<Int64, 1, Device::CPU> CleanedSeq2Indices(
	const TemplateLibrary::Vector<std::wstring>& Seq,
	const std::unordered_map<std::wstring, Int64>& Symbols
);

_D_Dragonian_Lib_Lib_Text_To_Speech_End