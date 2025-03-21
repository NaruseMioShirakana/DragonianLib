#include "../Nsf-Hifigan.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

Tensor<Float32, 3, Device::CPU> NsfHifigan::Forward(
	const Tensor<Float32, 4, Device::CPU>& _Mel,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0
) const
{
	if (!_F0.has_value())
		_D_Dragonian_Lib_Throw_Exception("F0 is required for this vocoder");
	_D_Dragonian_Lib_Rethrow_Block(return Inference(_Mel, _F0););
}

_D_Dragonian_Lib_Onnx_Vocoder_End