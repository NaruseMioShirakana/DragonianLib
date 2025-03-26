#include "../TTA2X.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

_D_Dragonian_Lib_Onnx_UnitsEncoder_Header

Tensor<Float32, 4, Device::CPU> TTA2X::Forward(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
	Tensor<Float32, 4, Device::CPU> Feats, Feats2, FeatsTTA;
	const auto MPaddingCount = _SamplingRate / 100;

	_D_Dragonian_Lib_Rethrow_Block(Feats = InferenceModel(
		_PCMData, _SamplingRate, _Mask
	).Evaluate(););

	Tensor<Float32, 3, Device::CPU> _MaskPadded;

	if (_Mask.has_value())
	{
		_D_Dragonian_Lib_Rethrow_Block(_MaskPadded = _Mask->get().Padding(
			{
				None,
				None,
				{MPaddingCount, 0}
			},
			PaddingType::Zero
		).Evaluate(););
	}

	if (_Mask.has_value())
		_D_Dragonian_Lib_Rethrow_Block(Feats2 = InferenceModel(
			_PCMData.Padding(
				{
					None,
					None,
					{MPaddingCount, 0}
				},
				PaddingType::Zero
			).Evaluate(),
			_SamplingRate,
			_MaskPadded
		).Evaluate(););
	else
		_D_Dragonian_Lib_Rethrow_Block(Feats2 = InferenceModel(
			_PCMData.Padding(
				{
					None,
					None,
					{MPaddingCount, 0}
				},
				PaddingType::Zero
			).Evaluate(),
			_SamplingRate,
			std::nullopt
		).Evaluate(););

	const auto PaddingCount = Feats2.Size(2) - Feats.Size(2);
	const auto BatchSize = Feats.Size(0);
	const auto Channels = Feats.Size(1);

	if (PaddingCount > 0)
		_D_Dragonian_Lib_Rethrow_Block(Feats = Feats.Padding(
			{
				None,
				None,
				{0, 1},
				None
			},
			PaddingType::Zero
		).Evaluate(););

	try
	{
		FeatsTTA = Functional::Cat(Feats2, Feats, 3).Evaluate().View(BatchSize, Channels, -1, GetUnitsDims());
		FeatsTTA = FeatsTTA[{":", ":", "1:"}];
		if (PaddingCount)
			FeatsTTA = FeatsTTA[{":", ":", { 0, -PaddingCount }}];
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Rethrow_Block(return FeatsTTA.Continuous().Evaluate(););
}

_D_Dragonian_Lib_Onnx_UnitsEncoder_End