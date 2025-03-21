#pragma once
#include "../ModelBase.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

namespace UnitsEncoder
{
	class Hubert : public LibSvcModule
	{
	public:
		Hubert() = delete;
		Hubert(
			const std::wstring& _HubertPath,
			const ExecutionProviders& _ExecutionProvider,
			unsigned _DeviceID,
			unsigned _ThreadCount
		);

		Hubert(
			const std::wstring& _HubertPath,
			const std::shared_ptr<DragonianLibOrtEnv>& _Env
		);

		TemplateLibrary::Vector<float> InferPCMData(
			TemplateLibrary::Vector<float>& _PCMData
		) const;

	private:
		std::shared_ptr<Ort::Session> _MyModel = nullptr;
		Int64 _MySamplingRate = 16000;
		Int64 _MyInputShape = 1;
		std::array<const char*, 2> _MyInputNames;
		std::array<const char*, 1> _MyOutputNames;
		bool _HasBatchDim = false;
	};
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End