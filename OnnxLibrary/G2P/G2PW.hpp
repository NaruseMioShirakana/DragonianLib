/**
 * @file G2PW.hpp
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
 * @brief G2PW Model
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once

#include "Libraries/G2P/CppPinYin.hpp"
#include "OnnxLibrary/Base/OrtBase.hpp"

_D_Dragonian_Lib_G2P_Header

struct G2PWModelHParams
{
	const void* Configs; // CppPinYinConfigs
	const wchar_t* ModelPath; // Path to the G2PW Model (const wchar_t*)
	const void* Enviroment; // Pointer to OnnxRuntime::OnnxRuntimeEnvironment (const OnnxRuntime::OnnxRuntimeEnvironment*)
	const void* Logger; // Pointer to DragonianLib::DLogger (const DragonianLib::DLogger*)
};

class G2PWModel : public CppPinYin, public OnnxRuntime::OnnxModelBase<G2PWModel>
{
public:
	G2PWModel(
		const void* Parameter = nullptr
	);
	~G2PWModel() override = default;

	std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) override;

	std::pair<Vector<std::wstring>, Vector<Int64>> Forward(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	);

	Vector<std::wstring> ConvertSegment(
		const std::wstring& Seg,
		const CppPinYinParameters& Parameters
	) override = 0; //TODO: Implement this function

	G2PWModel(const G2PWModel&) = default;
	G2PWModel(G2PWModel&&) = default;
	G2PWModel& operator=(const G2PWModel&) = default;
	G2PWModel& operator=(G2PWModel&&) = default;
};

_D_Dragonian_Lib_G2P_End