#include "Vector.h"
#include <string>
#include <filesystem>

_D_Dragonian_Lib_Template_Library_Space_Begin

void _Impl_Dragonian_Lib_Template_Library_Throw_Exception(const char* Message, const char* FILE, const char* FUN, int LINE)
{
	const std::string __DragonianLib__Message__ = Message;
	const std::string __DragonianLib__Message__Prefix__ =
		std::string("[In File: \"") + std::filesystem::path(FILE).filename().string() + "\", " +
		"Function: \"" + FUN + "\", " +
		"Line: " + std::to_string(LINE) + " ] ";
	if (__DragonianLib__Message__.substr(0, __DragonianLib__Message__Prefix__.length()) != __DragonianLib__Message__Prefix__)
		throw std::exception((__DragonianLib__Message__Prefix__ + __DragonianLib__Message__).c_str());
	throw std::exception(__DragonianLib__Message__.c_str());
}

_D_Dragonian_Lib_Template_Library_Space_End
