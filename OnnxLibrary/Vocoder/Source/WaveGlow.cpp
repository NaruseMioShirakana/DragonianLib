#include "../WaveGlow.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

Tensor<Float32, 3, Device::CPU> WaveGlow::Forward(
	const Tensor<Float32, 4, Device::CPU>& _Mel,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>>
) const
{
	_D_Dragonian_Lib_Rethrow_Block(return Inference(_Mel););
}

_D_Dragonian_Lib_Onnx_Vocoder_End