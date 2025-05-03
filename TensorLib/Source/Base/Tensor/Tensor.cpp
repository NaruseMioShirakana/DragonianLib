#include <regex>
#include "../../../Include/Base/Tensor/Tensor.h"

_D_Dragonian_Lib_Space_Begin

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

void SetTaskPoolSize(SizeType _Size)
{
	Operators::SetTaskPoolSize(_Size);
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

void Range::Parse(const std::string_view& _RangeArgs)
{
	static std::regex Begin_Step_End_Regex_Token(
		R"([ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*)"
	);
	static std::regex Begin_Step_End_Regex(
		R"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::regex Begin_End_Regex(
		R"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::regex End_Regex(
		R"([ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::regex Begin_Regex(
		R"([ ]*(-{0,1}\d+)[ ]*:[ ]*)"
	);
	static std::regex Value_Regex(
		R"([ ]*(-{0,1}\d+)[ ]*)"
	);
	
	std::match_results<decltype(_RangeArgs.cbegin())> _Match;
	if (_RangeArgs == ":")
		return;
	if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Value_Regex))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, End_Regex))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_Regex))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_Step_End_Regex))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

void Range::Parse(const std::wstring_view& _RangeArgs)
{
	static std::wregex Begin_Step_End_Regexw_Token(
		LR"([ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*(:){0,1}[ ]*([0-9]*)[ ]*)"
	);
	static std::wregex Begin_Step_End_Regexw(
		LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::wregex Begin_End_Regexw(
		LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::wregex End_Regexw(
		LR"([ ]*:[ ]*(-{0,1}\d+)[ ]*)"
	);
	static std::wregex Begin_Regexw(
		LR"([ ]*(-{0,1}\d+)[ ]*:[ ]*)"
	);
	static std::wregex Value_Regexw(
		LR"([ ]*(-{0,1}\d+)[ ]*)"
	);

	std::match_results<decltype(_RangeArgs.cbegin())> _Match;
	if (_RangeArgs == L":")
		return;
	if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Value_Regexw))
	{
		Step = std::stoll(_Match[1].str());
		Begin = Step;
		End = Step;
	}
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		End = std::stoll(_Match[2].str());
	}
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, End_Regexw))
		End = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_Regexw))
		Begin = std::stoll(_Match[1].str());
	else if (std::regex_match(_RangeArgs.cbegin(), _RangeArgs.cend(), _Match, Begin_Step_End_Regexw))
	{
		Begin = std::stoll(_Match[1].str());
		Step = std::stoll(_Match[2].str());
		End = std::stoll(_Match[3].str());
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Illegal Parameters!");
}

Range::Range(const char* _RangeArgs)
{
	Parse(_RangeArgs);
}

Range::Range(const wchar_t* _RangeArgs)
{
	Parse(_RangeArgs);
}

Range::Range(const std::string& _RangeArgs)
{
	Parse(_RangeArgs);
}

Range::Range(const std::wstring& _RangeArgs)
{
	Parse(_RangeArgs);
}

_D_Dragonian_Lib_Space_End