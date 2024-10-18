#include "Util/StringPreprocess.h"
#include <regex>
#ifdef _WIN32
#include <Windows.h>
#endif
namespace DragonianLib
{
	std::wregex newlinereg(L"\\n");
	std::wregex returnreg(L"\\r");
	std::wregex refreg(L"\"");

	std::string WideStringToUTF8(const std::wstring& input)
	{
#ifdef _WIN32
		std::vector<char> ByteString(input.length() * 6);
		WideCharToMultiByte(
			CP_UTF8,
			0,
			input.c_str(),
			int(input.length()),
			ByteString.data(),
			int(ByteString.size()),
			nullptr,
			nullptr
		);
		return ByteString.data();
#else
		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return converter.to_bytes(input);
#endif
	}

	std::string UnicodeToAnsi(const std::wstring& input)
	{
#ifdef _WIN32
		std::vector<char> ByteString(input.length() * 6);
		WideCharToMultiByte(
			CP_ACP,
			0,
			input.c_str(),
			int(input.length()),
			ByteString.data(),
			int(ByteString.size()),
			nullptr,
			nullptr
		);
		return ByteString.data();
#else
		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return converter.to_bytes(input);
#endif
	}

	std::wstring UTF8ToWideString(const std::string& input)
	{
#ifdef _WIN32
		std::vector<wchar_t> WideString(input.length() * 2);
		MultiByteToWideChar(
			CP_UTF8,
			0,
			input.c_str(),
			int(input.length()),
			WideString.data(),
			int(WideString.size())
		);
		return WideString.data();
#else
		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return converter.from_bytes(input);
#endif
	}

	std::wstring& replace_sp(std::wstring inp)
	{
		inp = std::regex_replace(inp, newlinereg, L"\\n");
		inp = std::regex_replace(inp, returnreg, L"\\r");
		inp = std::regex_replace(inp, refreg, L"\\\"");
		return inp;
	}

	std::wstring SerializeStringVector(const std::vector<std::string>& vector)
	{
		std::wstring vecstr = L"[";
		for (const auto& it : vector)
			if (!it.empty())
				vecstr += L'\"' + replace_sp(UTF8ToWideString(it)) + L"\", ";
		if (vecstr.length() > 2)
			vecstr = vecstr.substr(0, vecstr.length() - 2);
		vecstr += L']';
		return vecstr;
	}

	std::wstring SerializeStringVector(const std::vector<std::wstring>& vector)
	{
		std::wstring vecstr = L"[";
		for (const auto& it : vector)
			if (!it.empty())
				vecstr += L'\"' + replace_sp(it) + L"\", ";
		if (vecstr.length() > 2)
			vecstr = vecstr.substr(0, vecstr.length() - 2);
		vecstr += L']';
		return vecstr;
	}
}
