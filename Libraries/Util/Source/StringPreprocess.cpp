#include "Libraries/Util/StringPreprocess.h"
#include <regex>

#ifdef _WIN32
#include <Windows.h>
#else
#include <codecvt>
#endif

_D_Dragonian_Lib_Space_Begin

static const std::wregex _Valdef_Dragonian_Lib_New_Line_Regex___(L"\\n");
static const std::wregex _Valdef_Dragonian_Lib_Return_Regex___(L"\\r");
static const std::wregex _Valdef_Dragonian_Lib_Reference_Regex___(L"\"");
static const std::wregex _Valdef_Dragonian_Lib_Slash_Regex___(L"\\\\");

const std::string& WideStringToUTF8(const std::string& input)
{
	return input;
}

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

const std::string& UnicodeToAnsi(const std::string& input)
{
	return input;
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

const std::wstring& UTF8ToWideString(const std::wstring& input)
{
	return input;
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

std::wstring& ReplaceSpecialSymbol(std::wstring inp)
{
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_New_Line_Regex___, L"\\n");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Return_Regex___, L"\\r");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Reference_Regex___, L"\\\"");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Slash_Regex___, L"\\\\");
	return inp;
}

std::wstring SerializeStringVector(DragonianLibSTL::Vector<std::string>& vector)
{
	std::wstring vecstr = L"[";
	for (const auto& it : vector)
		if (!it.empty())
			vecstr += L'\"' + ReplaceSpecialSymbol(UTF8ToWideString(it)) + L"\", ";
	if (vecstr.length() > 2)
		vecstr = vecstr.substr(0, vecstr.length() - 2);
	vecstr += L']';
	return vecstr;
}

std::wstring SerializeStringVector(DragonianLibSTL::Vector<std::wstring>& vector)
{
	std::wstring vecstr = L"[";
	for (const auto& it : vector)
		if (!it.empty())
			vecstr += L'\"' + ReplaceSpecialSymbol(it) + L"\", ";
	if (vecstr.length() > 2)
		vecstr = vecstr.substr(0, vecstr.length() - 2);
	vecstr += L']';
	return vecstr;
}

std::wstring Number2Chinese(const std::wstring& _Number)
{
	std::wstring StrRtn;
	std::wstring InputStr = _Number;
	const size_t PIndex = InputStr.find(L'.');
	std::wstring IntegerStr, FractionStr;
	if (PIndex != std::wstring::npos)
	{
		IntegerStr = InputStr.substr(0, PIndex);
		FractionStr = InputStr.substr(PIndex + 1);
		while (!FractionStr.empty() && FractionStr.back() == L'0')
			FractionStr.pop_back();
	}
	else
		IntegerStr = std::move(InputStr);

	if (IntegerStr != L"0")
	{
		size_t MaxIntegerStrLength = IntegerStr.length();
		for (; MaxIntegerStrLength > 0; --MaxIntegerStrLength)
			if (IntegerStr[MaxIntegerStrLength - 1] != L'0')
				break;
		if (MaxIntegerStrLength < 1)
			MaxIntegerStrLength = 1;

		const auto DigitNum = IntegerStr.length();
		for (size_t i = 0; i < MaxIntegerStrLength; i++)
		{
			const auto NumberIndex = IntegerStr[i] - L'0';
			const auto DigitIndex = DigitNum - i - 1;
			if (0 == NumberIndex)
			{
				if ((i > 0 && L'0' == IntegerStr[i - 1]) || i == IntegerStr.length() - 1)
					continue;
				if (DigitIndex >= 4 && 0 == DigitIndex % 4)
					StrRtn += PreDefinedRegex::ChineseNumberDigit[DigitIndex];
				else
					StrRtn += PreDefinedRegex::ChineseNumber[NumberIndex];
			}
			else
			{
				StrRtn += PreDefinedRegex::ChineseNumber[NumberIndex];
				if (IntegerStr.length() == 2 && IntegerStr[0] == '1' && i == 0)
					StrRtn.erase(0);
				if (0 == DigitIndex % 4)
					StrRtn += PreDefinedRegex::ChineseNumberDigit[DigitIndex];
				else
					StrRtn += PreDefinedRegex::ChineseNumberDigit[DigitIndex % 4];
			}
		}
	}
	else
		StrRtn += L"零";

	if (!FractionStr.empty())
		StrRtn += L"点";
	for (const auto FractionI : FractionStr)
	{
		const auto NumberIndex = FractionI - L'0';
		StrRtn += PreDefinedRegex::ChineseNumber[NumberIndex];
	}
	return StrRtn;
}

UInt32 U16Word2Unicode(const std::wstring& input)
{
	if (input.empty())
		_D_Dragonian_Lib_Throw_Exception("Invalid UTF-16 sequence: too short.");
	if constexpr (sizeof(wchar_t) == 4)
		return static_cast<UInt32>(input[0]);
	else
	{
		if (input.length() > 2)
			_D_Dragonian_Lib_Throw_Exception("Invalid UTF-16 sequence: too long.");
		for (size_t i = 0; i < input.size(); ++i) {
			wchar_t ch = input[i];
			if (ch >= 0xD800 && ch <= 0xDBFF)
			{
				if (i + 1 < input.size()) 
				{
					wchar_t ch2 = input[i + 1];
					if (ch2 >= 0xDC00 && ch <= 0xDFFF)
						return ((ch - 0xD800) << 10) + (ch2 - 0xDC00) + 0x10000;
					_D_Dragonian_Lib_Throw_Exception("Invalid UTF-16 sequence: missing low surrogate");
				}
				_D_Dragonian_Lib_Throw_Exception("Invalid UTF-16 sequence: missing low surrogate");
			}
			return static_cast<UInt32>(ch);
		}
		_D_Dragonian_Lib_Throw_Exception("Invalid UTF-16 sequence: too short.");
	}
}

_D_Dragonian_Lib_Space_End
