/**
 * FileName: Logger.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <filesystem>
#include <mutex>

namespace DragonianLib {

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

