#include "../Register.hpp"
#include "../TTA2X.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_Onnx_UnitsEncoder_Header

std::unordered_map<std::wstring, Constructor> _GlobalUnitsEncoders;
std::vector<std::wstring> _GlobalUnitsEncoderList;

void RegisterUnitsEncoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (_GlobalUnitsEncoders.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"UnitsEncoderManager");
		return;
	}
	_GlobalUnitsEncoders[_PluginName] = _Constructor;
	_GlobalUnitsEncoderList.push_back(_PluginName);
}

UnitsEncoder New(
	const std::wstring& Name,
	const std::wstring& _Path,
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	const std::shared_ptr<Logger>& _Logger
)
{
	const auto fnpair = _GlobalUnitsEncoders.find(Name);
	try
	{
		if (fnpair != _GlobalUnitsEncoders.end())
			return fnpair->second(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable to find UnitsEncoder");
}


const std::vector<std::wstring>& GetUnitsEncoderList()
{
	return _GlobalUnitsEncoderList;
}

class Init
{
public:
	Init()
	{
		RegisterUnitsEncoder(
			L"HubertSoft",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"CNHubertSoftFish",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-768",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-768-l12",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"HubertBase-1024-l24",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hubert>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<ContentVec>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-256-l9",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<ContentVec>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-256-l12",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<ContentVec>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l9",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<ContentVec>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l12",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<ContentVec>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
		RegisterUnitsEncoder(
			L"ContentVec-768-l12-tta2x",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _UnitsDims, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<TTA2X>(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger);
			}
		);
	}
};

Init _Valdef_Init;

_D_Dragonian_Lib_Onnx_UnitsEncoder_End