/**
 * FileName: MJson.h
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
#include <string>
#include "MyTemplateLibrary/Vector.h"

struct yyjson_val;
struct yyjson_doc;

namespace DragonianLib
{
	/**
	 * @class MJsonValue
	 * @brief Implementation of JSON value
	 */
    class MJsonValue
    {
    public:
        template<typename T>
        using Vector = DragonianLibSTL::Vector<T>;

        MJsonValue() = default;
        MJsonValue(yyjson_val* _val) : _Ptr(_val) {}
        ~MJsonValue() = default;
        MJsonValue(MJsonValue&& _val) noexcept
        {
            _Ptr = _val._Ptr;
            _val._Ptr = nullptr;
        }
        MJsonValue(const MJsonValue& _val)
        {
            _Ptr = _val._Ptr;
        }
        MJsonValue& operator=(MJsonValue&& _val) noexcept
        {
            _Ptr = _val._Ptr;
            _val._Ptr = nullptr;
            return *this;
        }
        MJsonValue& operator=(const MJsonValue& _val)
        {
            _Ptr = _val._Ptr;
            return *this;
        }

        /**
         * @brief Check if the value is null
         * @return True if the value is null, false otherwise
         */
        [[nodiscard]] bool IsNull() const;

        /**
         * @brief Check if the value is a boolean
         * @return True if the value is a boolean, false otherwise
         */
        [[nodiscard]] bool IsBoolean() const;

        /**
         * @brief Check if the value is a boolean (alias for IsBoolean)
         * @return True if the value is a boolean, false otherwise
         */
        [[nodiscard]] bool IsBool() const;

        /**
         * @brief Check if the value is an integer
         * @return True if the value is an integer, false otherwise
         */
        [[nodiscard]] bool IsInt() const;

        /**
         * @brief Check if the value is a float
         * @return True if the value is a float, false otherwise
         */
        [[nodiscard]] bool IsFloat() const;

        /**
         * @brief Check if the value is a 64-bit integer
         * @return True if the value is a 64-bit integer, false otherwise
         */
        [[nodiscard]] bool IsInt64() const;

        /**
         * @brief Check if the value is a double
         * @return True if the value is a double, false otherwise
         */
        [[nodiscard]] bool IsDouble() const;

        /**
         * @brief Check if the value is a string
         * @return True if the value is a string, false otherwise
         */
        [[nodiscard]] bool IsString() const;

        /**
         * @brief Check if the value is an array
         * @return True if the value is an array, false otherwise
         */
        [[nodiscard]] bool IsArray() const;

        /**
         * @brief Get the boolean value
         * @return The boolean value
         */
        [[nodiscard]] bool GetBool() const;

        /**
         * @brief Get the boolean value (alias for GetBool)
         * @return The boolean value
         */
        [[nodiscard]] bool GetBoolean() const;

        /**
         * @brief Get the integer value
         * @return The integer value
         */
        [[nodiscard]] int GetInt() const;

        /**
         * @brief Get the 64-bit integer value
         * @return The 64-bit integer value
         */
        [[nodiscard]] int64_t GetInt64() const;

        /**
         * @brief Get the float value
         * @return The float value
         */
        [[nodiscard]] float GetFloat() const;

        /**
         * @brief Get the double value
         * @return The double value
         */
        [[nodiscard]] double GetDouble() const;

        /**
         * @brief Get the string value
         * @return The string value
         */
        [[nodiscard]] std::string GetString() const;

        /**
         * @brief Get the array value
         * @return The array value
         */
        [[nodiscard]] Vector<MJsonValue> GetArray() const;

        /**
         * @brief Get the size of the value
         * @return The size of the value
         */
        [[nodiscard]] size_t GetSize() const;

        /**
         * @brief Get the size of the value (alias for GetSize)
         * @return The size of the value
         */
        [[nodiscard]] size_t Size() const;

        /**
         * @brief Get the length of the string value
         * @return The length of the string value
         */
        [[nodiscard]] size_t GetStringLength() const;

        /**
         * @brief Get the value associated with the given key
         * @param _key The key to look up
         * @return The value associated with the key
         */
        [[nodiscard]] MJsonValue Get(const std::string& _key) const;

        /**
         * @brief Get the value associated with the given key (operator overload)
         * @param _key The key to look up
         * @return The value associated with the key
         */
        [[nodiscard]] MJsonValue operator[](const std::string& _key) const;

        /**
         * @brief Get the value at the given index (operator overload)
         * @param _idx The index to look up
         * @return The value at the index
         */
        [[nodiscard]] MJsonValue operator[](size_t _idx) const;

        /**
         * @brief Check if the value is empty
         * @return True if the value is empty, false otherwise
         */
        [[nodiscard]] bool Empty() const;

        /**
         * @brief Get the number of members in the value
         * @return The number of members in the value
         */
        [[nodiscard]] size_t GetMemberCount() const;

        /**
         * @brief Get the array of members in the value
         * @return The array of members in the value
         */
        [[nodiscard]] Vector<std::pair<std::string, MJsonValue>> GetMemberArray() const;

        /**
         * @brief Check if the value has a member with the given key
         * @param _key The key to look up
         * @return True if the value has a member with the key, false otherwise
         */
        [[nodiscard]] bool HasMember(const std::string& _key) const;

    protected:
        yyjson_val* _Ptr = nullptr;
    };

	/**
	 * @class MJsonDocument
	 * @brief Implementation of JSON document
	 */
    class MJsonDocument : public MJsonValue
    {
    public:
        MJsonDocument() = default;

        /**
         * @brief Construct a new MJsonDocument object from a file path
         * @param _path The file path
         */
        MJsonDocument(const wchar_t* _path);

        /**
         * @brief Construct a new MJsonDocument object from a string
         * @param _data The string data
         */
        MJsonDocument(const std::string& _data);

        ~MJsonDocument();

        MJsonDocument(MJsonDocument&& _Right) noexcept;

        MJsonDocument(const MJsonDocument& _Right) = delete;

        MJsonDocument& operator=(MJsonDocument&& _Right) noexcept;

        MJsonDocument& operator=(const MJsonDocument& _Right) = delete;

        /**
         * @brief Parse the JSON document from a string
         * @param _str The string data
         */
        void Parse(const std::string& _str);

    private:
        yyjson_doc* _document = nullptr;
    };
}

