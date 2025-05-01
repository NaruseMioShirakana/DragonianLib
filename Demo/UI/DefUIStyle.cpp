#include "DefUIStyle.h"
#include "MainWindow.h"
#include "DefControl.hpp"
#include "Libraries/MJson/MJson.h"
#include "Libraries/Util/StringPreprocess.h"

namespace SimpleF0Labeler
{
	std::wstring WindowTitle = L"SimpleLabeler";

	void CreateDefaultStyle(Mui::XML::MuiXML* xmlUI)
	{
		xmlUI->LoadDefaultStyle(true);
		xmlUI->Mgr()->LoadStyleList();
		xmlUI->AddStringList(L"MainWindowTitle", WindowTitle);
		auto mgr = xmlUI->Mgr();

		const auto LocalizationDocument = DragonianLib::MJson::MJsonDocument(
			DragonianLib::WideStringToUTF8(
				DragonianLib::GetCurrentFolder() +
				L"/localization.json"
			).c_str()
		);
		const auto LocalizationDict = LocalizationDocument.GetMemberArray();
		for (auto& [Key, Value] : LocalizationDict)
		{
			if (Key.empty() || !Value.IsString() || Value.Empty())
				continue;
			xmlUI->AddStringList(
				DragonianLib::UTF8ToWideString(Key),
				DragonianLib::UTF8ToWideString(Value.GetString())
			);
		}

		WndControls::SetLanguageXML(xmlUI);
	}
}