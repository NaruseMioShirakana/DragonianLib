#include <regex>
#include "../../../Include/Base/Tensor/Tensor.h"

_D_Dragonian_Lib_Space_Begin

std::regex _Valdef_Dragonian_Lib_Begin_Step_End_Regex_Token(R"([ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*)");
std::wregex _Valdef_Dragonian_Lib_Begin_Step_End_Regexw_Token(LR"([ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*)");
std::regex _Valdef_Dragonian_Lib_Begin_Step_End_Regex(R"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::wregex _Valdef_Dragonian_Lib_Begin_Step_End_Regexw(LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::regex _Valdef_Dragonian_Lib_Begin_End_Regex(R"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::wregex _Valdef_Dragonian_Lib_Begin_End_Regexw(LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::regex _Valdef_Dragonian_Lib_End_Regex(R"([ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::wregex _Valdef_Dragonian_Lib_End_Regexw(LR"([ ]*:[ ]*(-{0,1}\d+)[ ]*)");
std::regex _Valdef_Dragonian_Lib_Begin_Regex(R"([ ]*(-{0,1}\d+)[ ]*:[ ]*)");
std::wregex _Valdef_Dragonian_Lib_Begin_Regexw(LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*)");
std::regex _Valdef_Dragonian_Lib_Value_Regex(R"([ ]*(-{0,1}\d+)[ ]*)");
std::wregex _Valdef_Dragonian_Lib_Value_Regexw(LR"([ ]*(-{0,1}\d+)[ ]*)");

void SetRandomSeed(SizeType _Seed)
{
	Operators::GetThreadPool().SetRandomSeed(_Seed);
	Operators::GetRandomDeviceId() = 0;
}

void SetWorkerCount(SizeType _ThreadCount)
{
	Operators::GetThreadPool().Init(std::max(_ThreadCount, static_cast<SizeType>(0)));
	SetMaxTaskCountPerOperator(Operators::GetThreadPool().GetThreadCount() / 2);
}

void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount)
{
	
	Operators::SetMaxTaskCountPerOperator(std::max(_MaxTaskCount, static_cast<SizeType>(1)));
}

void EnableTimeLogger(bool _Enable)
{
	Operators::GetThreadPool().EnableTimeLogger(_Enable);
}

void EnableInstantRun(bool _Enable)
{
	Operators::GetThreadPool().EnableInstantRun(_Enable);
	Operators::SetInstantRunFlag(_Enable);
}

Range::Range(const char* _RangeArgs)
{
	std::cmatch _Match;
	if (strcmp(_RangeArgs, ":") == 0)
		return;
	if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Value_Regex))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_End_Regex))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_Regex))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_Step_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

Range::Range(const wchar_t* _RangeArgs)
{
	std::wcmatch _Match;
	if (wcscmp(_RangeArgs, L":") == 0)
		return;
	if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Value_Regexw))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_End_Regexw))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_Regexw))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs, _Match, _Valdef_Dragonian_Lib_Begin_Step_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

Range::Range(const std::string& _RangeArgs)
{
	std::cmatch _Match;
	if (strcmp(_RangeArgs.c_str(), ":") == 0)
		return;
	if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Value_Regex))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_End_Regex))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_Regex))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_Step_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

Range::Range(const std::wstring& _RangeArgs)
{
	std::wcmatch _Match;
	if (wcscmp(_RangeArgs.c_str(), L":") == 0)
		return;
	if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Value_Regexw))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_End_Regexw))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_Regexw))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.c_str(), _Match, _Valdef_Dragonian_Lib_Begin_Step_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

_D_Dragonian_Lib_Space_End