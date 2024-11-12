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
}

void BasicG2P::Release()
{

}

std::pair<Vector<std::wstring>, Vector<Int64>> BasicG2P::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	std::lock_guard Lock(_MyMutex);
	Vector<std::wstring> Result;
	auto PhonemeList = _MyConvert(
		_MyInstance,
		InputText.c_str(),
		LanguageID.c_str(),
		UserParameter
	);
	while (*PhonemeList)
	{
		Result.EmplaceBack(*PhonemeList);
		PhonemeList += 2;
	}
	return Result;
}

_D_Dragonian_Lib_G2P_End