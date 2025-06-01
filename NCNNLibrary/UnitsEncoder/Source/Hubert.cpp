#include "NCNNLibrary/UnitsEncoder/Hubert.hpp"
#include "NCNNLibrary/NCNNBase/Source/NCNNImpl.hpp"

_D_Dragonian_Lib_NCNN_UnitsEncoder_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_NCNN_Space GetDefaultLogger()->GetLoggerId() + L"::UnitsEncoder",
		_D_Dragonian_Lib_NCNN_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

HubertBase::HubertBase(
	const std::wstring& _Path,
	const NCNNOptions& Options,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : NCNNModel(_Path, Options, _AddCache, _Logger),
_MySamplingRate(_SamplingRate), _MyUnitsDims(_UnitsDims)
{

}

Tensor<Float32, 4, Device::CPU> HubertBase::InferenceModel(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
#ifdef _DEBUG
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
#endif

	if (_PCMData.Null())
		_D_Dragonian_Lib_Throw_Exception("PCMData is Null, please check the input tensor");

	if (m_NCNNNet->input_names().size() == 2)
	{
		const auto FrameAxis = _MyUnitsAxis == 2 ? 3 : 2;
		if (!_Mask.has_value() || _Mask.value().get().Null())
			_D_Dragonian_Lib_Throw_Exception("Mask is required");

		if (_Mask->get().Size(0) != _PCMData.Size(0))
			_D_Dragonian_Lib_Throw_Exception(
				"Batch/Channel of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(0)) +
				", got: " +
				std::to_string(_Mask->get().Size(0))
			);
		if (_Mask->get().Size(1) != _PCMData.Size(1))
			_D_Dragonian_Lib_Throw_Exception(
				"Channel/Batch of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(1)) +
				", got: " +
				std::to_string(_Mask->get().Size(1))
			);
		if (_Mask->get().Size(2) != _PCMData.Size(FrameAxis))
			_D_Dragonian_Lib_Throw_Exception(
				"Frames of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(FrameAxis)) +
				", got: " +
				std::to_string(_Mask->get().Size(2))
			);
	}

	Tensor<Float32, 3, Device::CPU> AudioCont = _PCMData.View();
	// ReSharper disable once CppEntityAssignedButNoRead
	Tensor<Float32, 3, Device::CPU> MaskCont;

	if (_SamplingRate != _MySamplingRate)
		_D_Dragonian_Lib_Rethrow_Block(
			AudioCont = AudioCont.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(2),
				IScale(double(_MySamplingRate) / double(_SamplingRate))
			).Evaluate();
		);
	
	auto Extractor = m_NCNNNet->create_extractor();

	_D_Dragonian_Lib_Rethrow_Block(AudioCont = ExtractorInput(0, Extractor, AudioCont););
	if (m_NCNNNet->input_names().size() == 2)
		_D_Dragonian_Lib_Rethrow_Block(MaskCont = ExtractorInput(1, Extractor, _Mask->get()););

	Tensor<Float32, 4, Device::CPU> Ret;
	_D_Dragonian_Lib_Rethrow_Block(
		Ret = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(
		L"Units Encoder Forward Inference With Audio Shape: [" +
		std::to_wstring(AudioCont.Shape(0)) + L", " +
		std::to_wstring(AudioCont.Shape(1)) + L", " +
		std::to_wstring(AudioCont.Shape(2)) + L"], Cost Time: " +
		std::to_wstring(
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - TimeBegin
			).count()
		) +
		L"ms"
	);
#endif

	return Ret;
}

Tensor<Float32, 4, Device::CPU> HubertBase::Forward(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
	_D_Dragonian_Lib_Rethrow_Block(return InferenceModel(_PCMData, _SamplingRate, _Mask););
}

_D_Dragonian_Lib_NCNN_UnitsEncoder_End