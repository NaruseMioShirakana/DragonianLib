#include "Util/Logger.h"
#include "Util/StringPreprocess.h"
#ifdef _WIN32
#include <Windows.h>
#else
#error
#endif

namespace DragonianLib
{
	Logger GlobalLogger;

	Logger::~Logger()
	{
	}

	Logger::Logger()
	{
	}

	Logger::Logger(logger_fn error_fn, logger_fn log_fn)
	{
		custom_logger_fn = true;
		cerror_fn = error_fn;
		cloggerfn = log_fn;
	}

	void Logger::log(const std::wstring& format)
	{
		std::lock_guard mtx(mx);
		if (custom_logger_fn)
		{
			cloggerfn(format.c_str(), nullptr);
			return;
		}
		fprintf_s(stdout, "%s\n", WideStringToUTF8(format).c_str());
	}

	void Logger::log(const char* format)
	{
		std::lock_guard mtx(mx);
		if (custom_logger_fn)
		{
			cloggerfn(nullptr, format);
			return;
		}
		fprintf_s(stdout, "%s\n", format);
	}

	void Logger::error(const std::wstring& format)
	{
		std::lock_guard mtx(mx);
		if (custom_logger_fn)
		{
			cloggerfn(format.c_str(), nullptr);
			cerror_fn(format.c_str(), nullptr);
			return;
		}
		fprintf_s(stdout, "[ERROR]%s\n", WideStringToUTF8(format).c_str());
	}

	void Logger::error(const char* format)
	{
		std::lock_guard mtx(mx);
		if (custom_logger_fn)
		{
			cloggerfn(nullptr, format);
			cerror_fn(nullptr, format);
			return;
		}
		fprintf_s(stdout, "[ERROR]%s\n", format);
	}

	void Logger::set_custom_logger(logger_fn error, logger_fn log)
	{
		custom_logger_fn = true;
		cerror_fn = error;
		cloggerfn = log;
	}

	Logger& GetLogger()
	{
		return GlobalLogger;
	}
}
