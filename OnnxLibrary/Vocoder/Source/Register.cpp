#include "OnnxLibrary/Vocoder/Register.hpp"
#include "OnnxLibrary/Vocoder/Vocoder.hpp"
#include "OnnxLibrary/Vocoder/WaveGlow.hpp"
#include "OnnxLibrary/Vocoder/Hifigan.hpp"
#include "OnnxLibrary/Vocoder/Nsf-Hifigan.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_Onnx_Vocoder_Header

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

void RegisterVocoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (GetRegister().contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"VocoderManager");
		return;
	}
	GetRegister()[_PluginName] = _Constructor;
	GetList().push_back(_PluginName);
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
	const auto fnpair = GetRegister().find(Name);
	try
	{
		if (fnpair != GetRegister().end())
			return fnpair->second(_Path, _Environment, _SamplingRate, _MelBins, _Logger);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable to find Vocoder");
}

[[maybe_unused]] class Init
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
} InitModule;  // NOLINT(misc-use-internal-linkage)

_D_Dragonian_Lib_Onnx_Vocoder_End