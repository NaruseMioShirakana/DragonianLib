#include "../Register.hpp"
#include "../Vocoder.hpp"
#include "../WaveGlow.hpp"
#include "../Hifigan.hpp"
#include "../Nsf-Hifigan.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_Onnx_Vocoder_Header

std::unordered_map<std::wstring, Constructor> _GlobalVocoders;
std::vector<std::wstring> _GlobalVocoderList;

void RegisterVocoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (_GlobalVocoders.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"VocoderManager");
		return;
	}
	_GlobalVocoders[_PluginName] = _Constructor;
	_GlobalVocoderList.push_back(_PluginName);
}


Vocoder New(
	const std::wstring& Name,
	const std::wstring& _Path,
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate,
	Int64 _MelBins,
	const std::shared_ptr<Logger>& _Logger
)
{
	const auto fnpair = _GlobalVocoders.find(Name);
	try
	{
		if (fnpair != _GlobalVocoders.end())
			return fnpair->second(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable to find Vocoder");
}

const std::vector<std::wstring>& GetVocoderList()
{
	return _GlobalVocoderList;
}

class Init
{
public:
	Init()
	{
		RegisterVocoder(
			L"WaveGlow",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _MelBins, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<WaveGlow>(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
			}
		);

		RegisterVocoder(
			L"Hifigan",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _MelBins, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hifigan>(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
			}
		);

		RegisterVocoder(
			L"Nsf-Hifigan",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _MelBins, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<NsfHifigan>(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
			}
		);

		RegisterVocoder(
			L"HiFi-GAN",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _MelBins, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<Hifigan>(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
			}
		);

		RegisterVocoder(
			L"Nsf-HiFi-GAN",
			[](const std::wstring& _Path, const OnnxRuntimeEnvironment& _Environment, Int64 _SamplingRate, Int64 _MelBins, const std::shared_ptr<Logger>& _Logger)
			{
				return std::make_shared<NsfHifigan>(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
			}
		);
	}
};

Init _Valdef_Init;

_D_Dragonian_Lib_Onnx_Vocoder_End