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
#include <unordered_map>

namespace DragonianLib
{
	namespace PreDefinedRegex
	{
		static constexpr wchar_t _Valdef_Regex_Chinese_And_Japanese[] = LR"([\u4E00-\u9FFF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF])";
		static constexpr wchar_t _Valdef_Regex_Chinese_And_Japanese_String[] = LR"([\u4E00-\u9FFF\u3040-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]+)";
		static constexpr wchar_t _Valdef_Regex_All_Symbol[] = L"[ !@#$%^&*()_+\\-=`~,./;'\\[\\]<>?:\"{}|\\\\。？！，、；：“”‘’『』「」（）〔〕【】─…·—～《》〈〉]";
		static constexpr wchar_t _Valdef_Regex_All_Symbol_Group[] = L"[ !@#$%^&*()_+\\-=`~,./;'\\[\\]<>?:\"{}|\\\\。？！，、；：“”‘’『』「」（）〔〕【】─…·—～《》〈〉]+";
	}

	namespace PreDefinedReplaceMap
	{
		static inline const std::unordered_map<std::wstring, std::wstring> _PUNCTUATION_MAP{
			{ L"：", L"," }, { L"；", L"," }, { L"，", L"," }, { L"。", L"." }, { L"！", L"!" }, { L"？", L"?" },
			{ L"·", L"," }, { L"、", L"," }, { L"...", L"…" }, { L"$", L"." }, { L"“", L"'" },
			{ L"”", L"'" }, { L"‘", L"'" }, { L"’", L"'" }, { L"（", L"'" }, { L"）", L"'" }, { L"(", L"'" },
			{ L")", L"'" }, { L"《", L"'" }, { L"》", L"'" }, { L"【", L"'" }, { L"】", L"'" }, { L"[", L"'" },
			{ L"]", L"'" }, { L"—", L"-" }, { L"～", L"-" }, { L"~", L"-" }, { L"「", L"'" }, { L"」", L"'" }
		};
		static inline const std::unordered_map<std::wstring, std::wstring> _ALPHASYMBOL_MAP{
			{L"#", L"シャープ"}, { L"%", L"パーセント" }, { L"&", L"アンド" }, { L"+", L"プラス" }, { L"-", L"マイナス" },
			{ L":", L"コロン" }, { L";", L"セミコロン" }, { L"<", L"小なり" }, { L"=", L"イコール" }, { L">", L"大なり" },
			{ L"@", L"アット" }, { L"a", L"エー" }, { L"b", L"ビー" }, { L"c", L"シー" }, { L"d", L"ディー" },
			{ L"e", L"イー" }, { L"f", L"エフ" }, { L"g", L"ジー" }, { L"h", L"エイチ" }, { L"i", L"アイ" },
			{ L"j", L"ジェー" }, { L"k", L"ケー" }, { L"l", L"エル" }, { L"m", L"エム" }, { L"n", L"エヌ" },
			{ L"o", L"オー" }, { L"p", L"ピー" }, { L"q", L"キュー" }, { L"r", L"アール" }, { L"s", L"エス" },
			{ L"t", L"ティー" }, { L"u", L"ユー" }, { L"v", L"ブイ" }, { L"w", L"ダブリュー" }, { L"x", L"エックス" },
			{ L"y", L"ワイ" }, { L"z", L"ゼット" }, { L"α", L"アルファ" }, { L"β", L"ベータ" }, { L"γ", L"ガンマ" },
			{ L"δ", L"デルタ" }, { L"ε", L"イプシロン" }, { L"ζ", L"ゼータ" }, { L"η", L"イータ" }, { L"θ", L"シータ" },
			{ L"ι", L"イオタ" }, { L"κ", L"カッパ" }, { L"λ", L"ラムダ" }, { L"μ", L"ミュー" }, { L"ν", L"ニュー" },
			{ L"ξ", L"クサイ" }, { L"ο", L"オミクロン" }, { L"π", L"パイ" }, { L"ρ", L"ロー" }, { L"σ", L"シグマ" },
			{ L"τ", L"タウ" }, { L"υ", L"ウプシロン" }, { L"φ", L"ファイ" },{ L"χ", L"カイ" }, { L"ψ", L"プサイ" },
			{ L"ω", L"オメガ", } };
		static inline const std::vector<std::pair<std::wstring, std::wstring>> _CURRENCY_MAP{
			{L"\\$", L"ドル"}, { L"¥", L"円" }, { L"£", L"ポンド" }, { L"€", L"ユーロ" }
		};
	}

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
