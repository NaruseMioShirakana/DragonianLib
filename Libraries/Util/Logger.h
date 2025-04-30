/**
 * @file Logger.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Base logger of DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Util/Util.h"

_D_Dragonian_Lib_Space_Begin

class Logger : public std::enable_shared_from_this<Logger>
{
public:
	using LoggerFunction = void(*)(const wchar_t* Message, unsigned Level);
	enum class LogLevel : uint8_t { Info, Warn, Error, None };

	virtual ~Logger() noexcept;
	virtual void Log(const std::wstring& _Message, LogLevel _Level = LogLevel::Info, const wchar_t* _NameSpace = nullptr) noexcept;

protected:
	LoggerFunction _MyLoggerFn;
	std::wstring _MyId = L"DragonianLib";
	LogLevel _MyLevel = LogLevel::Info;

public:
	Logger(std::wstring _LoggerId = L"DragonianLib", LogLevel _LogLevel = LogLevel::Info, LoggerFunction _LogFunction = nullptr) noexcept;
	Logger(const Logger& _Parent, const std::wstring& _NameSpace) noexcept;
	Logger(const Logger&) = default;
	Logger(Logger&&) = default;
	Logger& operator=(const Logger&) = default;
	Logger& operator=(Logger&&) = default;

	void SetLoggerId(const std::wstring& Id) noexcept;
	void SetLoggerLevel(LogLevel Level, bool WithChild = false) noexcept;
	void SetLoggerFunction(LoggerFunction Function, bool WithChild = false) noexcept;

	template <typename _ThisType>
	decltype(auto) GetLoggerId(this _ThisType&& _Self) noexcept
	{
		return std::forward<_ThisType>(_Self)._MyId;
	}

	LogLevel GetLoggerLevel() const noexcept { return _MyLevel; }

	void LogWarn(const std::wstring& _Message, const wchar_t* _NameSpace = nullptr) noexcept { Log(_Message, LogLevel::Warn, _NameSpace); }
	void LogMessage(const std::wstring& _Message, const wchar_t* _NameSpace = nullptr) noexcept { Log(_Message, LogLevel::Info, _NameSpace); }
	void LogInfo(const std::wstring& _Message, const wchar_t* _NameSpace = nullptr) noexcept { Log(_Message, LogLevel::Info, _NameSpace); }
	void LogError(const std::wstring& _Message, const wchar_t* _NameSpace = nullptr) noexcept { Log(_Message, LogLevel::Error, _NameSpace); }

private:
	mutable std::vector<Logger*> _MyChildLoggers;
};

using DLogger = std::shared_ptr<Logger>;

DLogger& GetDefaultLogger() noexcept;

_D_Dragonian_Lib_Space_End

