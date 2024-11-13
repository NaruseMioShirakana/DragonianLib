#include "../G2PPlugin.hpp"

_D_Dragonian_Lib_G2P_Header

BasicG2P::BasicG2P(const void* UserParameter, Plugin::Plugin PluginInp) : _MyPlugin(std::move(PluginInp))
{
	Construct(UserParameter);
}

BasicG2P::~BasicG2P()
{
	Destory();
}

void BasicG2P::Initialize(const void* Parameter)
{
	_MyInstance = _MyPlugin->GetInstance(Parameter);
	_MyConvert = (G2PApiType)_MyPlugin->GetFunction("Convert", true);
	_MyGetExtraInfo = (G2PGetExtraInfoType)_MyPlugin->GetFunction("GetExtraInfo");
}

void BasicG2P::Release()
{
	_MyPlugin->DestoryInstance(_MyInstance);
}

std::pair<Vector<std::wstring>, Vector<Int64>> BasicG2P::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	std::lock_guard Lock(_MyMutex);
	Vector<std::wstring> Result1;
	Vector<Int64> Result2;
	auto PhonemeList = _MyConvert(
		_MyInstance,
		InputText.c_str(),
		LanguageID.c_str(),
		UserParameter
	);
	while (wcscmp(PhonemeList->Phoneme, L"[EOS]") != 0 && PhonemeList->Tone != INT64_MAX)
	{
		Result1.EmplaceBack(PhonemeList->Phoneme);
		Result2.EmplaceBack(PhonemeList->Tone);
		++PhonemeList;
	}
	return { std::move(Result1), std::move(Result2) };
}

_D_Dragonian_Lib_G2P_End