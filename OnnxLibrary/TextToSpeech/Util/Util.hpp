/**
 * @file Util.hpp
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
 * @brief Util for TextToSpeech
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/Base/OrtBase.hpp"

#define _D_Dragonian_Lib_Lib_Text_To_Speech_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace Text2Speech \
	{

#define _D_Dragonian_Lib_Lib_Text_To_Speech_End \
    } \
	_D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Lib_Text_To_Speech_Space _D_Dragonian_Lib_Onnx_Runtime_Space Text2Speech::


_D_Dragonian_Lib_Lib_Text_To_Speech_Header

struct HParams
{
	std::unordered_map<std::wstring, std::wstring> ModelPaths;

	Int64 SamplingRate = 22050;

	std::unordered_map<std::wstring, std::wstring> Parameters;
};

struct TextData
{
    
};

struct Parameters
{
	Int64 SpeakerId = 0;

    Float32 NoiseScale = 0.3f;

    int64_t Seed = 114514;

	std::unordered_map<std::wstring, std::wstring> ExtraParameters;
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End
