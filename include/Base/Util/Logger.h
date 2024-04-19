#pragma once
#include <filesystem>
#include <mutex>
#include "Util/StringPreprocess.h"

namespace libsvc{

	std::wstring GetCurrentFolder(const std::wstring& defualt = L"");

	class Logger
	{
	public:
		using logger_fn = void(*)(const wchar_t*, const char*);
		Logger();
		~Logger();
		Logger(logger_fn error_fn, logger_fn log_fn);
		void log(const std::wstring&);
		void log(const char*);
		void error(const std::wstring&);
		void error(const char*);
	private:
		bool custom_logger_fn = false;
		std::filesystem::path cur_log_dir, logpath, errorpath;
		logger_fn cerror_fn = nullptr, cloggerfn = nullptr;
		FILE* log_file = nullptr, * error_file = nullptr;
		std::mutex mx;
	};

	Logger& GetLogger();
}

