#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include <functional>
#include "Libraries/F0Extractor/DioF0Extractor.hpp"
#include "Libraries/F0Extractor/HarvestF0Extractor.hpp"
#include "Libraries/F0Extractor/NetF0Predictors.hpp"
#include "Libraries/F0Extractor/PluginBasedF0Extractor.hpp"
#include "Libraries/Base.h"

_D_Dragonian_Lib_F0_Extractor_Header

std::vector<std::wstring> _GlobalF0ExtractorsList;
std::unordered_map<std::wstring, Constructor> _GlobalRegisteredF0Extractors;

void RegisterPlugin(
	const std::wstring& _PluginPath,
	const std::wstring& _PluginName
)
{
	if (_PluginPath.empty() || _PluginName.empty())
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Could not register plugin: " + _PluginName + L" at " + _PluginPath, L"F0ExtractorManager");
		return;
	}

	if (_GlobalRegisteredF0Extractors.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" at " + _PluginPath + L" already registered", L"F0ExtractorManager");
		return;
	}
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(_PluginPath);
		_GlobalRegisteredF0Extractors.emplace(
			_PluginName,
			[Plugin](const void* UserParameter) -> F0Extractor {
				return std::make_shared<PluginF0Extractor>(Plugin, UserParameter);
			}
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_GlobalF0ExtractorsList.emplace_back(_PluginName);
}

void RegisterF0Extractors(
	const std::wstring& _PluginRootDirectory
)
{
	if (_PluginRootDirectory.empty())
		return;
	std::filesystem::path PluginRootDirectory(_PluginRootDirectory);
	for (const auto& PluginDirectoryEntry : std::filesystem::directory_iterator(PluginRootDirectory))
	{
		if (PluginDirectoryEntry.is_regular_file())
		{
			const auto Extension = PluginDirectoryEntry.path().extension().wstring();
			if (Extension != L".dll" && Extension != L".so" && Extension != L".dylib")
				continue;
			const auto PluginName = PluginDirectoryEntry.path().stem().wstring();
			RegisterPlugin(PluginDirectoryEntry.path().wstring(), PluginName);
		}
		else if (PluginDirectoryEntry.is_directory())
		{
			const auto PluginName = PluginDirectoryEntry.path().filename().wstring();
			const auto PluginPath = PluginDirectoryEntry.path() / (PluginName + (_WIN32 ? L".dll" : L".so"));
			RegisterPlugin(PluginPath.wstring(), PluginName);
		}
	}
}

void RegisterF0Extractor(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (_GlobalRegisteredF0Extractors.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"F0ExtractorManager");
		return;
	}
	_GlobalRegisteredF0Extractors.emplace(_PluginName, _Constructor);
	_GlobalF0ExtractorsList.emplace_back(_PluginName);
}

F0Extractor New(
	const std::wstring& Name,
	const void* UserParameter
)
{
	const auto F0ExtractorIt = _GlobalRegisteredF0Extractors.find(Name);
	try
	{
		if (F0ExtractorIt != _GlobalRegisteredF0Extractors.end())
			return F0ExtractorIt->second(UserParameter);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available F0Extractor");
}

const std::vector<std::wstring>& GetF0ExtractorList()
{
	return _GlobalF0ExtractorsList;
}

struct Init
{
	Init()
	{
		_GlobalF0ExtractorsList.clear();
		_GlobalRegisteredF0Extractors.clear();

		RegisterF0Extractor(
			L"Dio",
			[](const void*) -> F0Extractor {
				return std::make_shared<DioF0Extractor>();
			}
		);
		RegisterF0Extractor(
			L"Harvest",
			[](const void*) -> F0Extractor {
				return std::make_shared<HarvestF0Extractor>();
			}
		);
		RegisterF0Extractors(GetCurrentFolder() + L"/Plugins/F0Extractor");
	}
};
Init _Valdef_Init;

_D_Dragonian_Lib_F0_Extractor_End
