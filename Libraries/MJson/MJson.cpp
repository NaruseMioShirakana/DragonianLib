#include "MJson.h"

#include "Base.h"
#include "yyjson.h"
namespace DragonianLib
{
	bool MJsonValue::IsNull() const
	{
		return yyjson_is_null(_Ptr);
	}
	bool MJsonValue::IsBoolean() const
	{
		return yyjson_is_bool(_Ptr);
	}
	bool MJsonValue::IsBool() const
	{
		return yyjson_is_bool(_Ptr);
	}
	bool MJsonValue::IsInt() const
	{
		return yyjson_is_num(_Ptr);
	}
	bool MJsonValue::IsFloat() const
	{
		return yyjson_is_num(_Ptr);
	}
	bool MJsonValue::IsInt64() const
	{
		return yyjson_is_num(_Ptr);
	}
	bool MJsonValue::IsDouble() const
	{
		return yyjson_is_num(_Ptr);
	}
	bool MJsonValue::IsString() const
	{
		return yyjson_is_str(_Ptr);
	}
	bool MJsonValue::IsArray() const
	{
		return yyjson_is_arr(_Ptr);
	}
	bool MJsonValue::GetBool() const
	{
		return yyjson_get_bool(_Ptr);
	}
	bool MJsonValue::GetBoolean() const
	{
		return yyjson_get_bool(_Ptr);
	}
	int MJsonValue::GetInt() const
	{
		return int(yyjson_get_num(_Ptr));
	}
	int64_t MJsonValue::GetInt64() const
	{
		return int64_t(yyjson_get_num(_Ptr));
	}
	float MJsonValue::GetFloat() const
	{
		return float(yyjson_get_num(_Ptr));
	}
	double MJsonValue::GetDouble() const
	{
		return yyjson_get_num(_Ptr);
	}
	std::string MJsonValue::GetString() const
	{
		if (const auto _str = yyjson_get_str(_Ptr))
			return _str;
		return "";
	}
	MJsonValue::Vector<MJsonValue> MJsonValue::GetArray() const
	{
		Vector<MJsonValue> _ret;
		if (!IsArray())
			return {};
		const auto _PArray = _Ptr;
		size_t idx, max;
		yyjson_val* _Object;
		yyjson_arr_foreach(_PArray, idx, max, _Object)
			_ret.EmplaceBack(_Object);
		return _ret;
	}
	size_t MJsonValue::GetSize() const
	{
		return yyjson_get_len(_Ptr);
	}
	size_t MJsonValue::Size() const
	{
		return yyjson_get_len(_Ptr);
	}
	size_t MJsonValue::GetStringLength() const
	{
		return yyjson_get_len(_Ptr);
	}
	MJsonValue MJsonValue::Get(const std::string& _key) const
	{
		return yyjson_obj_get(_Ptr, _key.c_str());
	}
	MJsonValue MJsonValue::operator[](const std::string& _key) const
	{
		return yyjson_obj_get(_Ptr, _key.c_str());
	}
	MJsonValue MJsonValue::operator[](size_t _idx) const
	{
		if (!IsArray())
			return _Ptr;
		const auto _max = yyjson_arr_size(_Ptr);
		const auto _val = yyjson_arr_get_first(_Ptr);
		return _idx < _max ? _val + _idx : _val + _max - 1;
	}
	bool MJsonValue::Empty() const
	{
		if (!IsArray() && !IsString())
			return true;
		auto _max = yyjson_arr_size(_Ptr);
		if (IsString()) _max = yyjson_get_len(_Ptr);
		return !_max;
	}
	size_t MJsonValue::GetMemberCount() const
	{
		return yyjson_obj_size(_Ptr);
	}
	MJsonValue::Vector<std::pair<std::string, MJsonValue>> MJsonValue::GetMemberArray() const
	{
		Vector<std::pair<std::string, MJsonValue>> ret;
		yyjson_val* key;
		yyjson_obj_iter iter = yyjson_obj_iter_with(_Ptr);
		while ((key = yyjson_obj_iter_next(&iter))) {
			const auto val = yyjson_obj_iter_get_val(key);
			ret.EmplaceBack(MJsonValue(key).GetString(), val);
		}
		return ret;
	}
	bool MJsonValue::HasMember(const std::string& _key) const
	{
		return yyjson_obj_get(_Ptr, _key.c_str());
	}

	MJsonDocument::MJsonDocument(const wchar_t* _path)
	{
		const auto file = DragonianLib::FileGuard(_path, L"rb");
		_document = yyjson_read_fp(file, YYJSON_READ_NOFLAG, nullptr, nullptr);
		if (!_document)
			throw std::exception("Json Parse Error!");
		_Ptr = yyjson_doc_get_root(_document);
	}
	MJsonDocument::MJsonDocument(const std::string& _data)
	{
		_document = yyjson_read(_data.c_str(), _data.length(), YYJSON_READ_NOFLAG);
		if (!_document)
			throw std::exception("Json Parse Error!");
		_Ptr = yyjson_doc_get_root(_document);
	}
	MJsonDocument::~MJsonDocument()
	{
		if (_document)
		{
			yyjson_doc_free(_document);
			_document = nullptr;
			_Ptr = nullptr;
		}
	}
	MJsonDocument::MJsonDocument(MJsonDocument&& _Right) noexcept
	{
		_document = _Right._document;
		_Right._document = nullptr;
		_Ptr = yyjson_doc_get_root(_document);
	}
	MJsonDocument& MJsonDocument::operator=(MJsonDocument&& _Right) noexcept
	{
		if (_document)
			yyjson_doc_free(_document);
		_document = _Right._document;
		_Right._document = nullptr;
		_Ptr = yyjson_doc_get_root(_document);
		return *this;
	}
	void MJsonDocument::Parse(const std::string& _str)
	{
		_document = yyjson_read(_str.c_str(), _str.length(), YYJSON_READ_NOFLAG);
		if (!_document)
			throw std::exception("Json Parse Error!");
		_Ptr = yyjson_doc_get_root(_document);
	}
}
