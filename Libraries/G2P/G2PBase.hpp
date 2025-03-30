/**
 * @file G2PBase.hpp
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
 * @brief Base class for Grapheme-to-Phoneme models in DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <mutex>
#include "Libraries/Base.h"
#include "Libraries/MyTemplateLibrary/Vector.h"
#define _D_Dragonian_Lib_G2P_Header _D_Dragonian_Lib_Space_Begin namespace G2P {
#define _D_Dragonian_Lib_G2P_End _D_Dragonian_Lib_Space_End }

_D_Dragonian_Lib_G2P_Header

using namespace DragonianLibSTL;

/**
 * @class G2PBase
 * @brief Base class for Grapheme-to-Phoneme models in DragonianLib
 */
class G2PBase
{
public:
	
	G2PBase() = default;
	virtual ~G2PBase() = default;

	/**
	 * @brief Convert text to phonemes
	 * @param InputText Input text
	 * @param LanguageID Language ID
	 * @param UserParameter User parameter
	 * @return Pair of phonemes and tones
	 */
	virtual std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) const = 0;

	/**
	 * @brief Get extra information
	 * @return Pair of lock and extra information
	 */
	virtual void* GetExtraInfo() const = 0;

protected:
	void Construct(const void* Parameter);
	void Destory();
	virtual void Initialize(const void* Parameter) = 0;
	virtual void Release() = 0;

public:
	G2PBase(const G2PBase&) = default;
	G2PBase(G2PBase&&) = default;
	G2PBase& operator=(G2PBase&&) noexcept = default;
	G2PBase& operator=(const G2PBase&) = default;
};

_D_Dragonian_Lib_G2P_End