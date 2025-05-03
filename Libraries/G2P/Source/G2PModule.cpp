#include "Libraries/G2P/G2PModule.hpp"
#include "Libraries/G2P/CppPinYin.hpp"

_D_Dragonian_Lib_G2P_Header

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

G2PModule New(
	const std::wstring& Name,
	const void* Parameter
)
{
	const auto G2PModuleIt = GetRegister().find(Name);
	try
	{
		if (G2PModuleIt != GetRegister().end())
			return G2PModuleIt->second(Parameter);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available G2PModule");
}

static void RegisterPlugin(
	const std::wstring& _PluginPath,
	const std::wstring& _PluginName
)
{
	if (_PluginPath.empty() || _PluginName.empty())
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Could not register plugin: " + _PluginName + L" at " + _PluginPath, L"G2PModules");
		return;
	}

	if (!exists(std::filesystem::path(_PluginPath)))
		return;

	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" at " + _PluginPath + L" already registered", L"G2PModules");
		return;
	}
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(_PluginPath);
		GetRegister().emplace(
			_PluginName,
			[Plugin](const void* UserParameter) -> G2PModule {
				return std::make_shared<BasicG2P>(UserParameter, Plugin);
			}
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	GetList().emplace_back(_PluginName);
}

void RegisterG2PModules(
	const std::wstring& _PluginRootDirectory
)
{
	if (_PluginRootDirectory.empty())
		return;
	std::filesystem::path PluginRootDirectory(_PluginRootDirectory);
	if (!std::filesystem::exists(PluginRootDirectory))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin root directory: " + _PluginRootDirectory + L" does not exist", L"G2PModules");
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

void RegisterG2PModule(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"G2PModules");
		return;
	}
	GetRegister().emplace(_PluginName, _Constructor);
	GetList().emplace_back(_PluginName);
}

extern void RegG2PW();

[[maybe_unused]] struct Init {
	Init()
	{
		RegisterG2PModules(GetCurrentFolder() + L"/Plugins/G2P");
		RegisterG2PModule(
			L"CppPinYin",
			[](const void* Parameter) -> G2PModule {
				return std::make_shared<CppPinYin>(Parameter);
			}
		);
		RegisterG2PModule(
			L"pypinyin",
			[](const void* Parameter) -> G2PModule {
				return std::make_shared<CppPinYin>(Parameter);
			}
		);
		RegG2PW();
	}
} InitModule;  // NOLINT(misc-use-internal-linkage)

_D_Dragonian_Lib_G2P_End