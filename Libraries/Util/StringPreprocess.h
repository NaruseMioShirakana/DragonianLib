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
#include "Libraries/MyTemplateLibrary/Vector.h"
#include <regex>

_D_Dragonian_Lib_Space_Begin

namespace PreDefinedRegex
{
	static inline auto _Valdef_Regex_Chinese_And_Japanese =
		std::wregex(LR"([\u4E00-\u9FFF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF])");
	static inline auto _Valdef_Regex_Chinese_And_Japanese_String =
		std::wregex(LR"([\u4E00-\u9FFF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]+)");
	static inline auto _Valdef_Regex_All_Symbol =
		std::wregex(L"[ !@#$%^&*()_+\\-=`~,./;'\\[\\]<>?:\"{}|\\\\。？！，、；：“”‘’『』「」（）〔〕【】─…·—～《》〈〉]");
	static inline auto _Valdef_Regex_All_Symbol_Group =
		std::wregex(L"[ !@#$%^&*()_+\\-=`~,./;'\\[\\]<>?:\"{}|\\\\。？！，、；：“”‘’『』「」（）〔〕【】─…·—～《》〈〉]+");
}

std::string WideStringToUTF8(const std::wstring& input);

std::string UnicodeToAnsi(const std::wstring& input);

std::wstring UTF8ToWideString(const std::string& input);

std::wstring SerializeStringVector(const DragonianLibSTL::Vector<std::string>& vector);

std::wstring SerializeStringVector(const DragonianLibSTL::Vector<std::wstring>& vector);

template <typename T>
std::wstring SerializeVector(const DragonianLibSTL::Vector<T>& vector)
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

_D_Dragonian_Lib_Space_End
