#include "yyjson.h"
#include "Libraries/Util/StringPreprocess.h"
#include "Libraries/MJson/MJson.h"

_D_Dragonian_Lib_MJson_Namespace_Begin

static void ValueDeleter(void*) {}
static void DocDeleter(void* _Pointer) { yyjson_doc_free((yyjson_doc*)_Pointer); }

MJsonValue::MJsonValue(void* _Object, std::shared_ptr<YYJsonDoc> _Doc) noexcept :
	_MyObject((YYJsonVal*)_Object, ValueDeleter), _MyDocument(std::move(_Doc))
{

}

bool MJsonValue::IsNull() const noexcept
{
	return yyjson_is_null((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsBoolean() const noexcept
{
	return yyjson_is_bool((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsBool() const noexcept
{
	return yyjson_is_bool((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsInt() const noexcept
{
	return yyjson_is_num((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsFloat() const noexcept
{
	return yyjson_is_num((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsInt64() const noexcept
{
	return yyjson_is_num((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsDouble() const noexcept
{
	return yyjson_is_num((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsString() const noexcept
{
	return yyjson_is_str((yyjson_val*)_MyObject.get());
}
bool MJsonValue::IsArray() const noexcept
{
	return yyjson_is_arr((yyjson_val*)_MyObject.get());
}
bool MJsonValue::GetBool() const noexcept
{
	return yyjson_get_bool((yyjson_val*)_MyObject.get());
}
bool MJsonValue::GetBoolean() const noexcept
{
	return yyjson_get_bool((yyjson_val*)_MyObject.get());
}
int MJsonValue::GetInt() const noexcept
{
	return int(yyjson_get_num((yyjson_val*)_MyObject.get()));
}
int64_t MJsonValue::GetInt64() const noexcept
{
	return int64_t(yyjson_get_num((yyjson_val*)_MyObject.get()));
}
float MJsonValue::GetFloat() const noexcept
{
	return float(yyjson_get_num((yyjson_val*)_MyObject.get()));
}
double MJsonValue::GetDouble() const noexcept
{
	return yyjson_get_num((yyjson_val*)_MyObject.get());
}
std::string MJsonValue::GetString() const noexcept
{
	if (const auto _str = yyjson_get_str((yyjson_val*)_MyObject.get()))
		return _str;
	return "";
}
MJsonValue::Vector<MJsonValue> MJsonValue::GetArray() const noexcept
{
	Vector<MJsonValue> Result;
	if (!IsArray())
		return {};
	const auto Array = (yyjson_val*)_MyObject.get();
	size_t Index, MaxIdx;
	yyjson_val* Object;
	yyjson_arr_foreach(Array, Index, MaxIdx, Object)
		Result.EmplaceBack(Object, _MyDocument);
	return Result;
}
size_t MJsonValue::GetSize() const noexcept
{
	return yyjson_get_len((yyjson_val*)_MyObject.get());
}
size_t MJsonValue::Size() const noexcept
{
	return yyjson_get_len((yyjson_val*)_MyObject.get());
}
size_t MJsonValue::GetStringLength() const noexcept
{
	return yyjson_get_len((yyjson_val*)_MyObject.get());
}
MJsonValue MJsonValue::Get(const std::string& _Key) const noexcept
{
	return {yyjson_obj_get((yyjson_val*)_MyObject.get(), _Key.c_str()), _MyDocument};
}
MJsonValue MJsonValue::operator[](const std::string& _Key) const noexcept
{
	return {yyjson_obj_get((yyjson_val*)_MyObject.get(), _Key.c_str()), _MyDocument};
}
MJsonValue MJsonValue::operator[](size_t _Index) const
{
	if (!IsArray())
		_D_Dragonian_Lib_Throw_Exception("Object is not an array!");
	_Index = std::min(_Index, yyjson_arr_size((yyjson_val*)_MyObject.get()) - 1);
	const auto First = yyjson_arr_get_first((yyjson_val*)_MyObject.get());
	return {First + _Index, _MyDocument};
}
bool MJsonValue::Empty() const noexcept
{
	if (!IsArray() && !IsString())
		return true;
	auto Size = yyjson_arr_size((yyjson_val*)_MyObject.get());
	if (IsString()) Size = yyjson_get_len((yyjson_val*)_MyObject.get());
	return !Size;
}
size_t MJsonValue::GetMemberCount() const noexcept
{
	return yyjson_obj_size((yyjson_val*)_MyObject.get());
}
MJsonValue::Vector<std::pair<std::string, MJsonValue>> MJsonValue::GetMemberArray() const noexcept
{
	Vector<std::pair<std::string, MJsonValue>> Result;
	yyjson_val* KeyValue;
	yyjson_obj_iter Iterator = yyjson_obj_iter_with((yyjson_val*)_MyObject.get());
	while ((KeyValue = yyjson_obj_iter_next(&Iterator))) {
		const auto Value = yyjson_obj_iter_get_val(KeyValue);
		Result.EmplaceBack(MJsonValue(KeyValue, _MyDocument).GetString(), MJsonValue{ Value, _MyDocument });
	}
	return Result;
}
bool MJsonValue::HasMember(const std::string& _Key) const noexcept
{
	return yyjson_obj_get((yyjson_val*)_MyObject.get(), _Key.c_str());
}

MJsonDocument::MJsonDocument(const char* _Path)
{
	const auto File = FileGuard(UTF8ToWideString(_Path), L"rb");
	_MyDocument = { (YYJsonDoc*)yyjson_read_fp(File, YYJSON_READ_NOFLAG, nullptr, nullptr), DocDeleter };
	if (!_MyDocument)
		throw std::exception("Json Parse Error!");
	_MyObject = { (YYJsonVal*)yyjson_doc_get_root((yyjson_doc*)_MyDocument.get()), ValueDeleter };
}

MJsonDocument::MJsonDocument(const wchar_t* _Path)
{
	const auto File = FileGuard(_Path, L"rb");
	_MyDocument = { (YYJsonDoc*)yyjson_read_fp(File, YYJSON_READ_NOFLAG, nullptr, nullptr), DocDeleter };
	if (!_MyDocument)
		throw std::exception("Json Parse Error!");
	_MyObject = { (YYJsonVal*)yyjson_doc_get_root((yyjson_doc*)_MyDocument.get()), ValueDeleter };
}

void MJsonDocument::Parse(const std::string& _StringData)
{
	_MyDocument = { (YYJsonDoc*)yyjson_read(_StringData.c_str(), _StringData.length(), YYJSON_READ_NOFLAG), DocDeleter };
	if (!_MyDocument)
		throw std::exception("Json Parse Error!");
	_MyObject = { (YYJsonVal*)yyjson_doc_get_root((yyjson_doc*)_MyDocument.get()), ValueDeleter };
}

_D_Dragonian_Lib_MJson_Namespace_End
