#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include <functional>
#include "Libraries/F0Extractor/DioF0Extractor.hpp"
#include "Libraries/F0Extractor/HarvestF0Extractor.hpp"
#include "Libraries/F0Extractor/NetF0Predictors.hpp"
#include "Libraries/F0Extractor/PluginBasedF0Extractor.hpp"
#include "Libraries/Base.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

_D_Dragonian_Lib_F0_Extractor_Header

using GetF0ExtractorFn = std::function<F0Extractor(const void*)>;
std::vector<std::wstring> F0ExtractorsList;
std::unordered_map<std::wstring, GetF0ExtractorFn> RegisteredF0Extractors;

F0Extractor GetF0Extractor(
	const std::wstring& Name,
	const void* UserParameter
)
{
	const auto F0ExtractorIt = RegisteredF0Extractors.find(Name);
	try
	{
		if (F0ExtractorIt != RegisteredF0Extractors.end())
			return F0ExtractorIt->second(UserParameter);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available F0Extractor");
}

void RegisterPlugin(const std::wstring& _PluginRootDirectory, const std::wstring& _PluginName)
{
	if (RegisteredF0Extractors.contains(_PluginName))
		return;
	const auto PluginPath = _PluginRootDirectory + L"\\" + _PluginName;
	const auto _PluginFileName = _PluginName.substr(0, _PluginName.find_last_of('.'));
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(PluginPath);
		RegisteredF0Extractors.emplace(_PluginFileName, [Plugin](const void* UserParameter) -> F0Extractor {
			return std::make_shared<PluginF0Extractor>(Plugin, UserParameter);
			});
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	F0ExtractorsList.emplace_back(_PluginFileName);
}

void RegisterF0Extractor(const std::wstring& _PluginRootDirectory)
{
	if (_PluginRootDirectory.empty())
		return;

#ifdef _WIN32
	WIN32_FIND_DATAW FindFileData;
	HANDLE hFind = FindFirstFileW((_PluginRootDirectory + L"\\*.dll").c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return;
	do
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue;
		if (FindFileData.cFileName[0] == L'.')
			continue;
		const auto PluginName = FindFileData.cFileName;
		try
		{
			RegisterPlugin(_PluginRootDirectory, PluginName);
		}
		catch (std::exception& e)
		{
			FindClose(hFind);
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	} while (FindNextFileW(hFind, &FindFileData));
	FindClose(hFind);
#else
	DIR* dir;
	struct dirent* ptr;
	if ((dir = opendir(_PluginRootDirectory.c_str())) == nullptr)
		return;

	while ((ptr = readdir(dir)) != nullptr)
	{
		if (ptr->d_type == DT_DIR)
			continue;
		const auto PluginName = ptr->d_name;
		try {
			RegisterPlugin(_PluginRootDirectory, PluginName);
		}
		catch (std::exception& e)
		{
			closedir(dir);
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	}
	closedir(dir);
#endif
}

const std::vector<std::wstring>& GetF0ExtractorList()
{
	return F0ExtractorsList;
}

struct Init
{
	Init()
	{
		F0ExtractorsList.clear();
		RegisteredF0Extractors.clear();
		F0ExtractorsList.emplace_back(L"Dio");
		RegisteredF0Extractors.emplace(L"Dio", [](const void* UserParameter) -> F0Extractor {
			return std::make_shared<DioF0Extractor>();
			});
		F0ExtractorsList.emplace_back(L"Harvest");
		RegisteredF0Extractors.emplace(L"Harvest", [](const void* UserParameter) -> F0Extractor {
			return std::make_shared<HarvestF0Extractor>();
			});
#ifdef DRAGONIANLIB_ONNXRT_LIB
		F0ExtractorsList.emplace_back(L"RMVPE");
		RegisteredF0Extractors.emplace(L"RMVPE", [](const void* UserParameter) -> F0Extractor {
			auto Params = (const NetF0ExtractorSetting*)UserParameter;
			const auto& OrtEnv = *(std::shared_ptr<DragonianLibOrtEnv>*)Params->OrtEnv;
			return std::make_shared<RMVPEF0Extractor>(Params->ModelPath, OrtEnv);
			});
		F0ExtractorsList.emplace_back(L"MELPE");
		RegisteredF0Extractors.emplace(L"MELPE", [](const void* UserParameter) -> F0Extractor {
			auto Params = (const NetF0ExtractorSetting*)UserParameter;
			const auto& OrtEnv = *(std::shared_ptr<DragonianLibOrtEnv>*)Params->OrtEnv;
			return std::make_shared<MELPEF0Extractor>(Params->ModelPath, OrtEnv);
			});
		F0ExtractorsList.emplace_back(L"FCPE");
		RegisteredF0Extractors.emplace(L"FCPE", [](const void* UserParameter) -> F0Extractor {
			auto Params = (const NetF0ExtractorSetting*)UserParameter;
			const auto& OrtEnv = *(std::shared_ptr<DragonianLibOrtEnv>*)Params->OrtEnv;
			return std::make_shared<MELPEF0Extractor>(Params->ModelPath, OrtEnv);
			});
#endif
		RegisterF0Extractor(GetCurrentFolder() + L"/Plugins/F0Extractor");
	}
};
Init _Valdef_Init;

_D_Dragonian_Lib_F0_Extractor_End
