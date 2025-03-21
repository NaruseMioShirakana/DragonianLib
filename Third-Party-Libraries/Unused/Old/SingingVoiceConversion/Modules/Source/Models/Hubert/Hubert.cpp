#include "../../../Header/Models/Hubert/Hubert.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

namespace UnitsEncoder
{
	Hubert::Hubert(
		const std::wstring& _HubertPath,
		const ExecutionProviders& _ExecutionProvider,
		unsigned _DeviceID,
		unsigned _ThreadCount
	) : LibSvcModule(_ExecutionProvider, _DeviceID, _ThreadCount)
	{
		try
		{
			HubertModel = RefOrtCachedModel(_HubertPath, *OrtApiEnv);
		}
		catch (const std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	}

	Hubert::Hubert(
		const std::wstring& _HubertPath,
		const std::shared_ptr<DragonianLibOrtEnv>& _Env
	): LibSvcModule(_Env)
	{
		try
		{
			HubertModel = RefOrtCachedModel(_HubertPath, *_Env);
		}
		catch (const std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	}

	TemplateLibrary::Vector<float> Hubert::InferPCMData(
		TemplateLibrary::Vector<float>& _PCMData,
		Int64 _SamplingRate,
		std::optional<std::reference_wrapper<TemplateLibrary::Vector<float>>> _Mask
	) const
	{
		const int64_t inputShape[3] = { 1i64,1i64,(int64_t)hubertin.Size() };

		std::array<Ort::Value, 1> InputTensors = {
			Ort::Value::CreateTensor(
				*MemoryInfo,
				_PCMData.Data(),
				_PCMData.Size(),
				1
			) };
		inputTensorshu.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, hubertin.Data(), hubertin.Size(), inputShape, 3));
		OrtTensors hubertOut;
	}

}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End