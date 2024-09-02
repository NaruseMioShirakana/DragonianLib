#pragma once
#include <string>
#include "MyTemplateLibrary/Vector.h"

struct yyjson_val;
struct yyjson_doc;

namespace DragonianLib
{
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
		[[nodiscard]] bool IsNull() const;
		[[nodiscard]] bool IsBoolean() const;
		[[nodiscard]] bool IsBool() const;
		[[nodiscard]] bool IsInt() const;
		[[nodiscard]] bool IsFloat() const;
		[[nodiscard]] bool IsInt64() const;
		[[nodiscard]] bool IsDouble() const;
		[[nodiscard]] bool IsString() const;
		[[nodiscard]] bool IsArray() const;
		[[nodiscard]] bool GetBool() const;
		[[nodiscard]] bool GetBoolean() const;
		[[nodiscard]] int GetInt() const;
		[[nodiscard]] int64_t GetInt64() const;
		[[nodiscard]] float GetFloat() const;
		[[nodiscard]] double GetDouble() const;
		[[nodiscard]] std::string GetString() const;
		[[nodiscard]] Vector<MJsonValue> GetArray() const;
		[[nodiscard]] size_t GetSize() const;
		[[nodiscard]] size_t Size() const;
		[[nodiscard]] size_t GetStringLength() const;
		[[nodiscard]] MJsonValue Get(const std::string& _key) const;
		[[nodiscard]] MJsonValue operator[](const std::string& _key) const;
		[[nodiscard]] MJsonValue operator[](size_t _idx) const;
		[[nodiscard]] bool Empty() const;
		[[nodiscard]] size_t GetMemberCount() const;
		[[nodiscard]] Vector<std::pair<std::string, MJsonValue>> GetMemberArray() const;
		[[nodiscard]] bool HasMember(const std::string& _key) const;
	protected:
		yyjson_val* _Ptr = nullptr;
	};

	class MJsonDocument : public MJsonValue
	{
	public:
		MJsonDocument() = default;
		MJsonDocument(const wchar_t* _path);
		MJsonDocument(const std::string& _data);
		~MJsonDocument();
		MJsonDocument(MJsonDocument&& _Right) noexcept;
		MJsonDocument(const MJsonDocument& _Right) = delete;
		MJsonDocument& operator=(MJsonDocument&& _Right) noexcept;
		MJsonDocument& operator=(const MJsonDocument& _Right) = delete;
		void Parse(const std::string& _str);

	private:
		yyjson_doc* _document = nullptr;
	};
}

/*

class MJson
{
public:
	MJson() = default;
	MJson(const char* _path)
	{
		const auto file = FileGuard(_path);
		_document = yyjson_read_fp(file, YYJSON_READ_NOFLAG, nullptr, nullptr);
		if (!_document)
			throw std::exception("Json Parse Error !");
		root = yyjson_doc_get_root(_document);
	}
	MJson(const std::string& _data, bool _read_from_string)
	{
		if (_read_from_string)
			_document = yyjson_read(_data.c_str(), _data.length(), YYJSON_READ_NOFLAG);
		else
		{
			const auto file = FileGuard(_data.c_str());
			_document = yyjson_read_fp(file, YYJSON_READ_NOFLAG, nullptr, nullptr);
		}
		if (!_document)
			throw std::exception("Json Parse Error !");
		root = yyjson_doc_get_root(_document);
	}
	~MJson()
	{
		if(_document)
		{
			yyjson_doc_free(_document);
			_document = nullptr;
			root = nullptr;
		}
	}
	MJson(MJson&& _Right) noexcept
	{
		_document = _Right._document;
		_Right._document = nullptr;
		root = yyjson_doc_get_root(_document);
	}
	MJson(const MJson& _Right) = delete;
	MJson& operator=(MJson&& _Right) noexcept
	{
		if (_document)
			yyjson_doc_free(_document);
		_document = _Right._document;
		_Right._document = nullptr;
		root = yyjson_doc_get_root(_document);
		return *this;
	}
	MJson& operator=(const MJson& _Right) = delete;
	void Parse(const std::string& _str)
	{
		_document = yyjson_read(_str.c_str(), _str.length(), YYJSON_READ_NOFLAG);
		if (!_document)
			throw std::exception("Json Parse Error !");
		root = yyjson_doc_get_root(_document);
	}
	[[nodiscard]] bool HasMember(const std::string& _key) const
	{
		return yyjson_obj_get(root, _key.c_str());
	}
	[[nodiscard]] MJsonValue Get(const std::string& _key) const
	{
		return yyjson_obj_get(root, _key.c_str());
	}
	[[nodiscard]] MJsonValue operator[](const std::string& _key) const
	{
		return yyjson_obj_get(root, _key.c_str());
	}
	[[nodiscard]] MJsonValue operator[](size_t _idx) const
	{
		if (MJsonValue(root).IsArray())
			return root;
		const auto _max = yyjson_arr_size(root);
		const auto _val = yyjson_arr_get_first(root);
		return _idx < _max ? _val + _idx : _val + _max - 1;
	}
	[[nodiscard]] bool HasParseError() const
	{
		return _document == nullptr;
	}
	[[nodiscard]] bool IsNull() const
	{
		return yyjson_is_null(root);
	}
	[[nodiscard]] bool IsBoolean() const
	{
		return yyjson_is_bool(root);
	}
	[[nodiscard]] bool IsBool() const
	{
		return yyjson_is_bool(root);
	}
	[[nodiscard]] bool IsInt() const
	{
		return yyjson_is_num(root);
	}
	[[nodiscard]] bool IsFloat() const
	{
		return yyjson_is_num(root);
	}
	[[nodiscard]] bool IsInt64() const
	{
		return yyjson_is_num(root);
	}
	[[nodiscard]] bool IsDouble() const
	{
		return yyjson_is_num(root);
	}
	[[nodiscard]] bool IsString() const
	{
		return yyjson_is_str(root);
	}
	[[nodiscard]] bool IsArray() const
	{
		return yyjson_is_arr(root);
	}
	[[nodiscard]] bool GetBool() const
	{
		return yyjson_get_bool(root);
	}
	[[nodiscard]] bool GetBoolean() const
	{
		return yyjson_get_bool(root);
	}
	[[nodiscard]] int GetInt() const
	{
		return int(yyjson_get_num(root));
	}
	[[nodiscard]] int64_t GetInt64() const
	{
		return int64_t(yyjson_get_num(root));
	}
	[[nodiscard]] float GetFloat() const
	{
		return float(yyjson_get_num(root));
	}
	[[nodiscard]] double GetDouble() const
	{
		return yyjson_get_num(root);
	}
	[[nodiscard]] std::string GetString() const
	{
		if (const auto _str = yyjson_get_str(root))
			return _str;
		return "";
	}
	[[nodiscard]] std::vector<MJsonValue> GetArray() const
	{
		std::vector<MJsonValue> _ret;
		if (!IsArray())
			return {};
		const auto _PArray = root;
		size_t idx, max;
		yyjson_val* _Object;
		yyjson_arr_foreach(_PArray, idx, max, _Object)
			_ret.emplace_back(_Object);
		return _ret;
	}
	[[nodiscard]] size_t GetSize() const
	{
		return yyjson_get_len(root);
	}
	[[nodiscard]] size_t Size() const
	{
		return yyjson_get_len(root);
	}
	[[nodiscard]] size_t GetStringLength() const
	{
		return yyjson_get_len(root);
	}
	[[nodiscard]] size_t GetMemberCount() const
	{
		return yyjson_obj_size(root);
	}
	[[nodiscard]] std::vector<std::pair<std::string,MJsonValue>> GetMemberArray() const
	{
		std::vector<std::pair<std::string, MJsonValue>> ret;
		yyjson_val* key;
		yyjson_obj_iter iter = yyjson_obj_iter_with(root);
		while ((key = yyjson_obj_iter_next(&iter))) {
			const auto val = yyjson_obj_iter_get_val(key);
			ret.emplace_back(MJsonValue(key).GetString(), val);
		}
		return ret;
	}
private:
	yyjson_doc* _document = nullptr;
	yyjson_val* root = nullptr;
};
*/
