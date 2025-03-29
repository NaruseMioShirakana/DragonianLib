#include "../Text2Speech.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::Text2Speech",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

Tensor<Int64, 1, Device::CPU> CleanedText2Indices(
	const std::wstring& Text,
	const std::unordered_map<std::wstring, Int64>& Symbols
)
{
	if (Text.empty())
		_D_Dragonian_Lib_Throw_Exception("Text Could Not Be Empty!");
	auto Indices = Tensor<Int64, 1, Device::CPU>::New({ static_cast<SizeType>(Text.size()) });
	auto Data = Indices.Data();

	for (size_t i = 0; i < Text.size(); ++i)
	{
		const auto Res = Symbols.find(std::wstring() + Text[i]);
		if (Res == Symbols.end())
			*Data++ = Symbols.at(L"UNK");
		else
			*Data++ = Res->second;
	}
	
	return Indices;
}

Tensor<Int64, 1, Device::CPU> CleanedSeq2Indices(
	const TemplateLibrary::Vector<std::wstring>& Seq,
	const std::unordered_map<std::wstring, Int64>& Symbols
)
{
	if (Seq.Empty())
		_D_Dragonian_Lib_Throw_Exception("Seq Could Not Be Empty!");
	auto Indices = Tensor<Int64, 1, Device::CPU>::New({ static_cast<SizeType>(Seq.Size()) });
	auto Data = Indices.Data();
	for (const auto& i : Seq)
	{
		const auto Res = Symbols.find(i);
		if (Res == Symbols.end())
			*Data++ = Symbols.at(L"UNK");
		else
			*Data++ = Res->second;
	}
	return Indices;
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End