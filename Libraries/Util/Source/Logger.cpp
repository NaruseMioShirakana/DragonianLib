#include "Libraries/Util/Logger.h"

_D_Dragonian_Lib_Space_Begin

Logger::~Logger() noexcept = default;

Logger::Logger(std::wstring _LoggerId, LogLevel _LogLevel, LoggerFunction _LogFunction) noexcept
	: _MyLoggerFn(_LogFunction), _MyId(std::move(_LoggerId)), _MyLevel(_LogLevel)
{

}

Logger::Logger(const Logger& _Parent, const std::wstring& _NameSpace) noexcept
	: _MyLoggerFn(_Parent._MyLoggerFn), _MyId(_Parent._MyId + L"::" + _NameSpace), _MyLevel(_Parent._MyLevel)
{
		
}

void Logger::Log(const std::wstring& _Message, LogLevel _Level, const wchar_t* _NameSpace) noexcept
{
	if (_Level < _MyLevel || _Level == LogLevel::None)
		return;
	if (_MyLoggerFn)
		_MyLoggerFn(_Message.c_str(), static_cast<unsigned>(_Level));
	else
	{
		std::wstring Prefix = _Level == LogLevel::Info ? L"[Info; @" : _Level == LogLevel::Warn ? L"[Warn; @" : L"[Error; @";
		Prefix += _MyId;
		if (_NameSpace)
		{
			Prefix += L"::";
			Prefix += _NameSpace;
		}
		Prefix += L"]: "; Prefix += _Message;
		wprintf(L"%ls\n", Prefix.c_str());
	}
}

DLogger& GetDefaultLogger() noexcept
{
	static std::shared_ptr<Logger> DefaultLogger = std::make_shared<Logger>();
	return DefaultLogger;
}

_D_Dragonian_Lib_Space_End
