#include "UI/MainWindow.h"

namespace App
{
	Mui::MiaoUI m_engine;
	auto constexpr m_reskey = L"12345678";

	bool Application(std::wstring& errinfo, std::vector<std::wstring> cmdList)
	{
		if (!m_engine.InitEngine(errinfo))
			return false;

		if(!m_engine.AddResourcePath(std::wstring(DragonianLib::GetCurrentFolder() + L"\\MVSResource.dmres"), m_reskey))
		{
			errinfo = L"资源文件加载失败!";
			return false;
		}

		if(!UI::CreateMainWindow(m_engine, std::move(cmdList)))
		{
			errinfo = L"初始化窗口失败!";
			return false;
		}

		UI::MainEventLoop();

		return true;
	}
}

#ifdef _WIN32

int APIENTRY main()
{
	const std::locale loc = std::locale::global(std::locale(std::locale(), "", LC_CTYPE));
	std::locale::global(loc);

	//获取命令行
	std::vector<std::wstring> args;

	std::wstring errinfo;

	if(!App::Application(errinfo, std::move(args)))
	{
		MessageBoxW(nullptr, (L"初始化应用程序失败！错误信息:\n" + errinfo).c_str(), L"error", MB_ICONERROR);
		return -1;
	}

	return 0;
}

#else
#error __TODO__
#endif