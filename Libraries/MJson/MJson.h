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
#include <memory>
#include <string>
#include "Libraries/MyTemplateLibrary/Vector.h"

#define _D_Dragonian_Lib_MJson_Namespace_Begin _D_Dragonian_Lib_Space_Begin namespace MJson {
#define _D_Dragonian_Lib_MJson_Namespace_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_MJson_Namespace_Begin

struct YYJsonVal;
struct YYJsonDoc;

/**
 * @class MJsonValue
 * @brief Implementation of JSON value
 */
class MJsonValue
{
public:
    template<typename T>
    using Vector = DragonianLibSTL::Vector<T>;

    MJsonValue() noexcept = default;
    MJsonValue(void* _Object, std::shared_ptr<YYJsonDoc> _Doc) noexcept;
    ~MJsonValue() noexcept = default;
    MJsonValue(MJsonValue&&) noexcept = default;
    MJsonValue(const MJsonValue&) noexcept = default;
    MJsonValue& operator=(MJsonValue&&) noexcept = default;
    MJsonValue& operator=(const MJsonValue&) noexcept = default;

    /**
     * @brief Check if the value is null
     * @return True if the value is null, false otherwise
     */
    [[nodiscard]] bool IsNull() const noexcept;

    /**
     * @brief Check if the value is a boolean
     * @return True if the value is a boolean, false otherwise
     */
    [[nodiscard]] bool IsBoolean() const noexcept;

    /**
     * @brief Check if the value is a boolean (alias for IsBoolean)
     * @return True if the value is a boolean, false otherwise
     */
    [[nodiscard]] bool IsBool() const noexcept;

    /**
     * @brief Check if the value is an integer
     * @return True if the value is an integer, false otherwise
     */
    [[nodiscard]] bool IsInt() const noexcept;

    /**
     * @brief Check if the value is a float
     * @return True if the value is a float, false otherwise
     */
    [[nodiscard]] bool IsFloat() const noexcept;

    /**
     * @brief Check if the value is a 64-bit integer
     * @return True if the value is a 64-bit integer, false otherwise
     */
    [[nodiscard]] bool IsInt64() const noexcept;

    /**
     * @brief Check if the value is a double
     * @return True if the value is a double, false otherwise
     */
    [[nodiscard]] bool IsDouble() const noexcept;

    /**
     * @brief Check if the value is a string
     * @return True if the value is a string, false otherwise
     */
    [[nodiscard]] bool IsString() const noexcept;

    /**
     * @brief Check if the value is an array
     * @return True if the value is an array, false otherwise
     */
    [[nodiscard]] bool IsArray() const noexcept;

    /**
     * @brief Get the boolean value
     * @return The boolean value
     */
    [[nodiscard]] bool GetBool() const noexcept;

    /**
     * @brief Get the boolean value (alias for GetBool)
     * @return The boolean value
     */
    [[nodiscard]] bool GetBoolean() const noexcept;

    /**
     * @brief Get the integer value
     * @return The integer value
     */
    [[nodiscard]] int GetInt() const noexcept;

    /**
     * @brief Get the 64-bit integer value
     * @return The 64-bit integer value
     */
    [[nodiscard]] int64_t GetInt64() const noexcept;

    /**
     * @brief Get the float value
     * @return The float value
     */
    [[nodiscard]] float GetFloat() const noexcept;

    /**
     * @brief Get the double value
     * @return The double value
     */
    [[nodiscard]] double GetDouble() const noexcept;

    /**
     * @brief Get the string value
     * @return The string value
     */
    [[nodiscard]] std::string GetString() const noexcept;

    /**
     * @brief Get the array value
     * @return The array value
     */
    [[nodiscard]] Vector<MJsonValue> GetArray() const noexcept;

    /**
     * @brief Get the size of the value
     * @return The size of the value
     */
    [[nodiscard]] size_t GetSize() const noexcept;

    /**
     * @brief Get the size of the value (alias for GetSize)
     * @return The size of the value
     */
    [[nodiscard]] size_t Size() const noexcept;

    /**
     * @brief Get the length of the string value
     * @return The length of the string value
     */
    [[nodiscard]] size_t GetStringLength() const noexcept;

    /**
     * @brief Get the value associated with the given key
     * @param _Key The key to look up
     * @return The value associated with the key
     */
    [[nodiscard]] MJsonValue Get(const std::string& _Key) const noexcept;

    /**
     * @brief Get the value associated with the given key (operator overload)
     * @param _Key The key to look up
     * @return The value associated with the key
     */
    [[nodiscard]] MJsonValue operator[](const std::string& _Key) const noexcept;

    /**
     * @brief Get the value at the given index (operator overload)
     * @param _Index The index to look up
     * @return The value at the index
     */
    [[nodiscard]] MJsonValue operator[](size_t _Index) const;

    /**
     * @brief Check if the value is empty
     * @return True if the value is empty, false otherwise
     */
    [[nodiscard]] bool Empty() const noexcept;

    /**
     * @brief Get the number of members in the value
     * @return The number of members in the value
     */
    [[nodiscard]] size_t GetMemberCount() const noexcept;

    /**
     * @brief Get the array of members in the value
     * @return The array of members in the value
     */
    [[nodiscard]] Vector<std::pair<std::string, MJsonValue>> GetMemberArray() const noexcept;

    /**
     * @brief Check if the value has a member with the given key
     * @param _Key The key to look up
     * @return True if the value has a member with the key, false otherwise
     */
    [[nodiscard]] bool HasMember(const std::string& _Key) const noexcept;

protected:
    std::shared_ptr<YYJsonVal> _MyObject = nullptr;
	std::shared_ptr<YYJsonDoc> _MyDocument = nullptr;
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
     * @param _Path The file path
     */
    MJsonDocument(const wchar_t* _Path);

    /**
     * @brief Construct a new MJsonDocument object from a string
     * @param _StringData The string data
     */
    MJsonDocument(const std::string& _StringData);

    /**
     * @brief Parse the JSON document from a string
     * @param _StringData The string data
     */
    void Parse(const std::string& _StringData);

    ~MJsonDocument() = default;
    MJsonDocument(MJsonDocument&& _Right) noexcept = default;
    MJsonDocument(const MJsonDocument& _Right) = default;
    MJsonDocument& operator=(MJsonDocument&& _Right) noexcept = default;
    MJsonDocument& operator=(const MJsonDocument& _Right) = default;

};

_D_Dragonian_Lib_MJson_Namespace_End

