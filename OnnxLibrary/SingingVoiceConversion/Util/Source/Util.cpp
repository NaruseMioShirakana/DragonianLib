#include "../Util.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::SingingVoiceConversion",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

void SliceDatas::Clear() noexcept
{
	OrtValues.clear();
	DlibTuple.clear();
}

void SliceDatas::Emplace(
	std::pair<Ort::Value, std::shared_ptr<DlibValue>>&& _InputTensor
)
{
	auto Ten = std::move(_InputTensor);
	OrtValues.emplace_back(std::move(Ten.first));
	DlibTuple.emplace_back(std::move(Ten.second));
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End