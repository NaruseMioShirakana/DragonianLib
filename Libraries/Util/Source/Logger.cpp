#include "Libraries/Util/Logger.h"

namespace DragonianLib
{
	Logger GlobalLogger;

	Logger::~Logger() = default;

	Logger::Logger() = default;

	Logger::Logger(LoggerFunction Function)
	{
		LoggerFn_ = Function;
	}

	void Logger::Log(LogLevel Level, const wchar_t* Message, const wchar_t* Id)
	{
		if (Level < Level_)
			return;
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

	void Logger::Message(const wchar_t* Message)
	{
		std::lock_guard Lock(Mutex_);
		if (LoggerFn_)
			LoggerFn_(static_cast<unsigned>(LogLevel::None), Message, L"");
		else
			printf("%ls\n", Message);
	}

	Logger& Logger::operator<<(const wchar_t* Message)
	{
		Log(LogLevel::Info, Message, Id_.c_str());
		return *this;
	}

	Logger& Logger::operator<<(const std::wstring& Message)
	{
		Log(LogLevel::Info, Message.c_str(), Id_.c_str());
		return *this;
	}

	Logger& GetLogger()
	{
		return GlobalLogger;
	}

	void SetLoggerId(const wchar_t* Id)
	{
		GlobalLogger.SetLoggerId(Id);
	}

	void SetLoggerLevel(LogLevel Level)
	{
		GlobalLogger.SetLoggerLevel(Level);
	}

	void SetLoggerFunction(Logger::LoggerFunction Function)
	{
		GlobalLogger.SetLoggerFunction(Function);
	}

	void LogInfo(const wchar_t* Message)
	{
		GlobalLogger.Log(LogLevel::Info, Message, GlobalLogger.GetLoggerId().c_str());
	}

	void LogWarn(const wchar_t* Message)
	{
		GlobalLogger.Log(LogLevel::Warn, Message, GlobalLogger.GetLoggerId().c_str());
	}

	void LogError(const wchar_t* Message)
	{
		GlobalLogger.Log(LogLevel::Error, Message, GlobalLogger.GetLoggerId().c_str());
	}

	void LogMessage(const wchar_t* Message)
	{
		GlobalLogger.Message(Message);
	}

	void LogInfo(const std::wstring& Message)
	{
		GlobalLogger.Log(LogLevel::Info, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
	}

	void LogWarn(const std::wstring& Message)
	{
		GlobalLogger.Log(LogLevel::Warn, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
	}

	void LogError(const std::wstring& Message)
	{
		GlobalLogger.Log(LogLevel::Error, Message.c_str(), GlobalLogger.GetLoggerId().c_str());
	}

	void LogMessage(const std::wstring& Message)
	{
		GlobalLogger.Message(Message.c_str());
	}
}
