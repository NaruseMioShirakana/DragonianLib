#include "UI/MainWindow.h"

namespace App
{
    constexpr auto m_reskey = L"12345678";

	static bool Application(std::wstring& errinfo, std::vector<std::wstring> cmdList)
	{
        static auto m_engine = Mui::MiaoUI::CreateInstance();

		if (!m_engine->InitEngine(errinfo))
			return false;

		if(!m_engine->AddResource(std::wstring(DragonianLib::GetCurrentFolder() + L"\\MVSResource.dmres"), m_reskey))
		{
			errinfo = L"资源文件加载失败!";
			return false;
		}

		if(!UI::CreateMainWindow(*m_engine, std::move(cmdList)))
		{
			errinfo = L"初始化窗口失败!";
			return false;
		}

		UI::MainEventLoop();

		return true;
	}
}

#ifdef _WIN32

#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#include <fstream>

// ReSharper disable once CppParameterMayBeConstPtrOrRef
static void DumpStackTrace(EXCEPTION_POINTERS* pExceptionPointers)
{
    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();

    // 初始化调试帮助库
    SymInitialize(hProcess, nullptr, TRUE);

    // 创建堆栈帧
    CONTEXT* context = pExceptionPointers->ContextRecord;
    STACKFRAME64 stackFrame = {};
    DWORD machineType = IMAGE_FILE_MACHINE_I386;

#ifdef _M_X64
    machineType = IMAGE_FILE_MACHINE_AMD64;
    stackFrame.AddrPC.Offset = context->Rip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context->Rbp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context->Rsp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#elif _M_IX86
    machineType = IMAGE_FILE_MACHINE_I386;
    stackFrame.AddrPC.Offset = context->Eip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context->Ebp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context->Esp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#endif

    std::ofstream logFile("crash_dump.txt");
    logFile << "Call Stack:\n";

    // 遍历堆栈
    while (StackWalk64(machineType, hProcess, hThread, &stackFrame, context, nullptr,
        SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
    {
        DWORD64 address = stackFrame.AddrPC.Offset;
        if (address == 0)
            break;

        // 获取符号信息
        char symbolBuffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)] = {};
        SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolBuffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;

        if (SymFromAddr(hProcess, address, nullptr, symbol))
        {
            logFile << symbol->Name << " - 0x" << std::hex << symbol->Address << "\n";
        }
        else
        {
            logFile << "Unknown function at 0x" << std::hex << address << "\n";
        }
    }

    logFile.close();
    SymCleanup(hProcess);
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
static LONG WINAPI ExceptionHandler([[jetbrains::has_side_effects]] EXCEPTION_POINTERS* pExceptionPointers)
{
    DumpStackTrace(pExceptionPointers);
    return EXCEPTION_EXECUTE_HANDLER;
}

int APIENTRY main()
{
    SetUnhandledExceptionFilter(ExceptionHandler);

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