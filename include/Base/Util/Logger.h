#pragma once
#include <filesystem>
#include <mutex>

namespace libsvc{

	std::wstring GetCurrentFolder(const std::wstring& defualt = L"");

	class Logger
	{
	public:
		using logger_fn = void(*)(const wchar_t*, const char*);
		Logger();
		~Logger();
		Logger(logger_fn error_fn, logger_fn log_fn);
		Logger(const Logger&) = delete;
		Logger(Logger&&) = delete;
		Logger& operator=(const Logger&) = delete;
		Logger& operator=(Logger&&) = delete;
		void log(const std::wstring&);
		void log(const char*);
		void error(const std::wstring&);
		void error(const char*);
		void set_custom_logger(logger_fn error, logger_fn log);
	private:
		bool custom_logger_fn = false;
		logger_fn cerror_fn = nullptr, cloggerfn = nullptr;
		std::mutex mx;
	};

	Logger& GetLogger();
}

