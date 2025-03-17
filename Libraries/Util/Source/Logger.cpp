#include "Libraries/Util/Logger.h"

_D_Dragonian_Lib_Space_Begin

Logger GlobalLogger;

Logger::~Logger() noexcept = default;

Logger::Logger() noexcept = default;

Logger::Logger(LoggerFunction Function) noexcept
{
	LoggerFn_ = Function;
}

void Logger::Log(LogLevel Level, const wchar_t* Message, const wchar_t* Id) noexcept
{
	if (Level < Level_)
		return;
	if (!Id)
		Id = Id_.c_str();
	std::lock_guard Lock(Mutex_);
	if (LoggerFn_)
		LoggerFn_(static_cast<unsigned>(Level), Message, Id);
	else
	{
		std::wstring Prefix = Level == LogLevel::Info ? L"[Info; @" : Level == LogLevel::Warn ? L"[Warn; @" : L"[Error; @";
		Prefix += Id;
		Prefix += L"]: ";
		Prefix += Message;
		printf("%ls\n", Prefix.c_str());
	}
}

void Logger::Message(const wchar_t* Message) noexcept
{
	std::lock_guard Lock(Mutex_);
	if (LoggerFn_)
		LoggerFn_(static_cast<unsigned>(LogLevel::None), Message, L"");
	else
		printf("%ls\n", Message);
}

void Logger::Log(LogLevel Level, const std::wstring& Message, const std::wstring& Id) noexcept
{
	if (Id.empty())
		Log(Level, Message.c_str(), Id_.c_str());
	else
		Log(Level, Message.c_str(), Id.c_str());
}

Logger& Logger::operator<<(const wchar_t* Message) noexcept
{
	Log(LogLevel::Info, Message, Id_.c_str());
	return *this;
}

Logger& Logger::operator<<(const std::wstring& Message) noexcept
{
	Log(LogLevel::Info, Message.c_str(), Id_.c_str());
	return *this;
}

Logger& GetLogger() noexcept
{
	return GlobalLogger;
}

void SetLoggerId(const wchar_t* Id) noexcept
{
	GlobalLogger.SetLoggerId(Id);
}

void SetLoggerLevel(LogLevel Level) noexcept
{
	GlobalLogger.SetLoggerLevel(Level);
}

void SetLoggerFunction(Logger::LoggerFunction Function) noexcept
{
	GlobalLogger.SetLoggerFunction(Function);
}

void LogInfo(const wchar_t* Message) noexcept
{
	GlobalLogger.Log(LogLevel::Info, Message, GlobalLogger.GetLoggerId().c_str());
}

void LogWarn(const wchar_t* Message) noexcept
{
	GlobalLogger.Log(LogLevel::Warn, Message, GlobalLogger.GetLoggerId().c_str());
}

void LogError(const wchar_t* Message) noexcept
{
	GlobalLogger.Log(LogLevel::Error, Message, GlobalLogger.GetLoggerId().c_str());
}

void LogMessage(const wchar_t* Message) noexcept
{
	GlobalLogger.Message(Message);
}

void LogInfo(const std::wstring& Message) noexcept
{
	GlobalLogger.Log(LogLevel::Info, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
}

void LogWarn(const std::wstring& Message) noexcept
{
	GlobalLogger.Log(LogLevel::Warn, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
}

void LogError(const std::wstring& Message) noexcept
{
	GlobalLogger.Log(LogLevel::Error, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
}

void LogMessage(const std::wstring& Message) noexcept
{
	GlobalLogger.Message(Message.c_str());
}

_D_Dragonian_Lib_Space_End
