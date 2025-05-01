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
	_Parent.EmplaceChild(this);
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

void Logger::SetLoggerId(const std::wstring& Id) noexcept
{
	_MyId = Id;
}

void Logger::SetLoggerLevel(LogLevel Level, bool WithChild) noexcept
{
	std::lock_guard lg(_MyMutex);
	_MyLevel = Level;
	if (WithChild)
		for (auto& Child : _MyChildLoggers)
			Child->SetLoggerLevel(Level);
}

void Logger::SetLoggerFunction(LoggerFunction Function, bool WithChild) noexcept
{
	std::lock_guard lg(_MyMutex);
	_MyLoggerFn = Function;
	if (WithChild)
		for (auto& Child : _MyChildLoggers)
			Child->SetLoggerFunction(Function);
}

DLogger& GetDefaultLogger() noexcept
{
	static std::shared_ptr<Logger> DefaultLogger = std::make_shared<Logger>();
	return DefaultLogger;
}

void Logger::EmplaceChild(Logger* _Child) const
{
	if (_Child == this)
		return;
	std::lock_guard lg(_MyMutex);
	_MyChildLoggers.emplace_back(_Child);
}

_D_Dragonian_Lib_Space_End
