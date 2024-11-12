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
#include <mutex>

namespace DragonianLib {

	enum class LogLevel
	{
		Info,
		Warn,
		Error,
		None
	};

	class Logger
	{
	public:
		using LoggerFunction = void(*)(unsigned Level, const wchar_t* Message, const wchar_t* Id);
		Logger();
		~Logger();
		Logger(LoggerFunction Function);
		Logger(const Logger&) = delete;
		Logger(Logger&&) = delete;
		Logger& operator=(const Logger&) = delete;
		Logger& operator=(Logger&&) = delete;
		void Log(LogLevel Level, const wchar_t* Message, const wchar_t* Id);
		void Message(const wchar_t* Message);
		Logger& operator<<(const wchar_t* Message);
		Logger& operator<<(const std::wstring& Message);
		void SetLoggerId(const wchar_t* Id) { Id_ = Id; }
		void SetLoggerLevel(LogLevel Level) { Level_ = Level; }
		void SetLoggerFunction(LoggerFunction Function) { LoggerFn_ = Function; }
		std::wstring& GetLoggerId() { return Id_; }
	private:
		LoggerFunction LoggerFn_;
		std::mutex Mutex_;
		std::wstring Id_ = L"DragonianLib";
		LogLevel Level_ = LogLevel::Info;
	};

	Logger& GetLogger();
	void SetLoggerId(const wchar_t* Id);
	void SetLoggerLevel(LogLevel Level);
	void SetLoggerFunction(Logger::LoggerFunction Function);
	void LogInfo(const wchar_t* Message);
	void LogWarn(const wchar_t* Message);
	void LogError(const wchar_t* Message);
	void LogMessage(const wchar_t* Message);
	void LogInfo(const std::wstring& Message);
	void LogWarn(const std::wstring& Message);
	void LogError(const std::wstring& Message);
	void LogMessage(const std::wstring& Message);
}

