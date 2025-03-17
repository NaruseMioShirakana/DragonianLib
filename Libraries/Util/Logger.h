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
#include "Util.h"

_D_Dragonian_Lib_Space_Begin

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

	virtual ~Logger() noexcept;
	virtual void Log(LogLevel Level, const wchar_t* Message, const wchar_t* Id = nullptr) noexcept;
	virtual void Message(const wchar_t* Message) noexcept;
	
private:
	LoggerFunction LoggerFn_;
	std::mutex Mutex_;
	std::wstring Id_ = L"DragonianLib";
	LogLevel Level_ = LogLevel::Info;

public:
	Logger() noexcept;
	Logger(LoggerFunction Function) noexcept;
	Logger(const Logger&) = delete;
	Logger(Logger&&) = delete;
	Logger& operator=(const Logger&) = delete;
	Logger& operator=(Logger&&) = delete;
	void Log(LogLevel Level, const std::wstring& Message, const std::wstring& Id = L"") noexcept;
	Logger& operator<<(const wchar_t* Message) noexcept;
	Logger& operator<<(const std::wstring& Message) noexcept;
	void SetLoggerId(const wchar_t* Id) noexcept { Id_ = Id; }
	void SetLoggerLevel(LogLevel Level) noexcept { Level_ = Level; }
	void SetLoggerFunction(LoggerFunction Function) noexcept { LoggerFn_ = Function; }
	std::wstring& GetLoggerId() noexcept { return Id_; }
};

Logger& GetLogger() noexcept;
void SetLoggerId(const wchar_t* Id) noexcept;
void SetLoggerLevel(LogLevel Level) noexcept;
void SetLoggerFunction(Logger::LoggerFunction Function) noexcept;
void LogInfo(const wchar_t* Message) noexcept;
void LogWarn(const wchar_t* Message) noexcept;
void LogError(const wchar_t* Message) noexcept;
void LogMessage(const wchar_t* Message) noexcept;
void LogInfo(const std::wstring& Message) noexcept;
void LogWarn(const std::wstring& Message) noexcept;
void LogError(const std::wstring& Message) noexcept;
void LogMessage(const std::wstring& Message) noexcept;

_D_Dragonian_Lib_Space_End

