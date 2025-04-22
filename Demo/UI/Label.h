#pragma once
#include "framework.h"

namespace App
{
	//应用程序版本号
	auto const m_version = L"0.0.1";
	auto const m_versionCore = L"0.0.1";//Core版本号

	//运行应用程序主函数
	extern bool Application(std::wstring& errinfo, std::vector<std::wstring> cmdList);
}