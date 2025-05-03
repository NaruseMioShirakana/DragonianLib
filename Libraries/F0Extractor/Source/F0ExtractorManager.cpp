#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include "Libraries/F0Extractor/DioF0Extractor.hpp"
#include "Libraries/F0Extractor/HarvestF0Extractor.hpp"
#include "Libraries/F0Extractor/PluginBasedF0Extractor.hpp"
#include "Libraries/Base.h"
#include <functional>

_D_Dragonian_Lib_F0_Extractor_Header

static auto& GetRegister()
{
	static std::unordered_map<std::wstring, Constructor> Register;
	return Register;
}

std::vector<std::wstring>& GetList()
{
	static std::vector<std::wstring> GlobalList;
	return GlobalList;
}

static void RegisterPlugin(
	const std::wstring& _PluginPath,
	const std::wstring& _PluginName
)
{
	if (_PluginPath.empty() || _PluginName.empty())
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Could not register plugin: " + _PluginName + L" at " + _PluginPath, L"F0ExtractorManager");
		return;
	}

	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" at " + _PluginPath + L" already registered", L"F0ExtractorManager");
		return;
	}
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(_PluginPath);
		GetRegister().emplace(
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
	GetList().emplace_back(_PluginName);
}

void RegisterF0Extractors(
	const std::wstring& _PluginRootDirectory
)
{
	if (_PluginRootDirectory.empty())
		return;
	std::filesystem::path PluginRootDirectory(_PluginRootDirectory);
	if (!std::filesystem::exists(PluginRootDirectory))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin root directory: " + _PluginRootDirectory + L" does not exist", L"F0ExtractorManager");
		return;
	}
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
	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"F0ExtractorManager");
		return;
	}
	GetRegister().emplace(_PluginName, _Constructor);
	GetList().emplace_back(_PluginName);
}

F0Extractor New(
	const std::wstring& Name,
	const void* UserParameter
)
{
	const auto F0ExtractorIt = GetRegister().find(Name);
	try
	{
		if (F0ExtractorIt != GetRegister().end())
			return F0ExtractorIt->second(UserParameter);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available F0Extractor");
}

#ifdef DRAGONIANLIB_ONNXRT_LIB
extern void InitNetPE();
#endif

[[maybe_unused]] struct Init
{
	Init()
	{
		GetList().clear();
		GetRegister().clear();

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
#ifdef DRAGONIANLIB_ONNXRT_LIB
		InitNetPE();
#endif
	}
} InitModule;  // NOLINT(misc-use-internal-linkage)

_D_Dragonian_Lib_F0_Extractor_End
