#include "NCNNLibrary/UnitsEncoder/Register.hpp"
#include "NCNNLibrary/UnitsEncoder/TTA2X.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_NCNN_UnitsEncoder_Header

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

void RegisterUnitsEncoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"UnitsEncoderManager");
		return;
	}
	GetRegister()[_PluginName] = _Constructor;
	GetList().push_back(_PluginName);
}

UnitsEncoder New(
	const std::wstring& Name,
	const std::wstring& _Path,
	const NCNNOptions& Options,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
)
{
	const auto fnpair = GetRegister().find(Name);
	try
	{
		if (fnpair != GetRegister().end())
			return fnpair->second(
				_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
			);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable to find UnitsEncoder");
}

[[maybe_unused]] class Init
{
public:
	Init()
	{
		RegisterUnitsEncoder(
			L"HubertSoft",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"CNHubertSoftFish",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-768",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-768-l12",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-1024-l24",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<Hubert>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<ContentVec>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-256-l9",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<ContentVec>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-256-l12",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<ContentVec>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l9",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<ContentVec>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l12",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<ContentVec>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l12-tta2x",
			[](
				const std::wstring& _Path,
				const NCNNOptions& Options,
				Int64 _SamplingRate,
				Int64 _UnitsDims,
				bool _AddCache,
				const std::shared_ptr<Logger>& _Logger
				)
			{
				return std::make_shared<TTA2X>(
					_Path, Options, _SamplingRate, _UnitsDims, _AddCache, _Logger
				);
			}
		);
	}
} InitModule;  // NOLINT(misc-use-internal-linkage)

_D_Dragonian_Lib_NCNN_UnitsEncoder_End