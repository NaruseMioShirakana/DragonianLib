/**
 * FileName: StringPreprocess.h
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
*/

#pragma once
#include <vector>
#include <string>

namespace DragonianLib
{
	std::string WideStringToUTF8(const std::wstring& input);

	std::string UnicodeToAnsi(const std::wstring& input);

	std::wstring UTF8ToWideString(const std::string& input);

	std::wstring SerializeStringVector(const std::vector<std::string>& vector);

	std::wstring SerializeStringVector(const std::vector<std::wstring>& vector);

	template <typename T>
	std::wstring SerializeVector(const std::vector<T>& vector)
	{
		std::wstring vecstr = L"[";
		for (const auto& it : vector)
		{
			std::wstring TmpStr = std::to_wstring(it);
			if ((std::is_same_v<T, float> || std::is_same_v<T, double>) && TmpStr.find(L'.') != std::string::npos)
			{
				while (TmpStr.back() == L'0')
					TmpStr.pop_back();
				if (TmpStr.back() == L'.')
					TmpStr += L"0";
			}
			vecstr += TmpStr + L", ";
		}
		if (vecstr.length() > 2)
			vecstr = vecstr.substr(0, vecstr.length() - 2);
		vecstr += L']';
		return vecstr;
	}
}