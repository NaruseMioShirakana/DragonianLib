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