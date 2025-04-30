#include "Libraries/MyTemplateLibrary/Vector.h"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

_D_Dragonian_Lib_Template_Library_Space_End

_D_Dragonian_Lib_Space_Begin

static std::wstring& ReplaceSpecialSymbol(std::wstring inp)
{
	static const std::wregex _Valdef_Dragonian_Lib_New_Line_Regex___(L"\\n");
	static const std::wregex _Valdef_Dragonian_Lib_Return_Regex___(L"\\r");
	static const std::wregex _Valdef_Dragonian_Lib_Reference_Regex___(L"\"");
	static const std::wregex _Valdef_Dragonian_Lib_Slash_Regex___(L"\\\\");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_New_Line_Regex___, L"\\n");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Return_Regex___, L"\\r");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Reference_Regex___, L"\\\"");
	inp = std::regex_replace(inp, _Valdef_Dragonian_Lib_Slash_Regex___, L"\\\\");
	return inp;
}

std::wstring SerializeStringVector(const DragonianLibSTL::Vector<std::string>& vector)
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

std::wstring SerializeStringVector(const DragonianLibSTL::Vector<std::wstring>& vector)
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

_D_Dragonian_Lib_Space_End