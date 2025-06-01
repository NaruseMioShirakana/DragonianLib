#pragma once
#include "NCNNLibrary/NCNNBase/NCNNBase.h"
#include "ncnn/net.h"

_D_Dragonian_NCNN_Lib_Space_Header

template <typename ValueType, size_t Rank>
auto ExtractorInput(
	int __Ind,
	ncnn::Extractor& __Ext,
	const Tensor<ValueType, Rank, Device::CPU>& __InpRaw
)
	requires((sizeof(ValueType) == 4 || sizeof(ValueType) == 2 || sizeof(ValueType) == 1) && Rank > 0 && Rank < 5)
{
	auto __Inp = __InpRaw.Contiguous().Evaluate();
	char blob_name[] = { 'i', 'n',char(__Ind + '0'),'\0' };
	if constexpr (Rank == 1)
		__Ext.input(blob_name, ncnn::Mat(
			static_cast<int>(__Inp.Size(-1)),
			__Inp.Data(),
			sizeof(ValueType)
		));
	else if constexpr (Rank == 2)
		__Ext.input(blob_name, ncnn::Mat(
			static_cast<int>(__Inp.Size(-1)),
			static_cast<int>(__Inp.Size(-2)),
			__Inp.Data(),
			sizeof(ValueType)
		));
	else if constexpr (Rank == 3)
		__Ext.input(blob_name, ncnn::Mat(
			static_cast<int>(__Inp.Size(-1)),
			static_cast<int>(__Inp.Size(-2)),
			static_cast<int>(__Inp.Size(-3)),
			__Inp.Data(),
			sizeof(ValueType)
		));
	else if constexpr (Rank == 4)
		__Ext.input(blob_name, ncnn::Mat(
			static_cast<int>(__Inp.Size(-1)),
			static_cast<int>(__Inp.Size(-2)),
			static_cast<int>(__Inp.Size(-3)),
			static_cast<int>(__Inp.Size(-4)),
			__Inp.Data(),
			sizeof(ValueType)
		));
	return __Inp;
}

template <typename ValueType, size_t Rank>
auto ExtractorOutput(
	int __Ind,
	ncnn::Extractor& __Ext
)
	requires((sizeof(ValueType) == 4 || sizeof(ValueType) == 2 || sizeof(ValueType) == 1) && Rank > 0 && Rank < 5)
{
	constexpr auto __InpRnk = Rank;
	constexpr auto __InpEs = sizeof(ValueType);
	ncnn::Mat __Out;
	char blob_name[] = { 'o', 'u', 't',char(__Ind + '0'),'\0' };
	if (__Ext.extract(blob_name, __Out))
		_D_Dragonian_Lib_Throw_Exception("Error when running model!");
	if (__InpEs != __Out.elemsize)
		_D_Dragonian_Lib_Throw_Exception("Type mismatch!");

	auto __Data = __Out.data;
	auto __Shape = __Out.shape();
	auto __RetSz = __Shape.c * __Shape.d * __Shape.h * __Shape.w;

	if constexpr (__InpRnk == 1)
		return Functional::FromShared(
			Dimensions{ __Shape.w },
			// ReSharper disable once CppLambdaCaptureNeverUsed
			std::shared_ptr<void>(__Data, [__Out](void*) {}),
			__RetSz
		);
	else if constexpr (__InpRnk == 2)
		return Functional::FromShared(
			Dimensions{ __Shape.h, __Shape.w },
			// ReSharper disable once CppLambdaCaptureNeverUsed
			std::shared_ptr<void>(__Data, [__Out](void*) {}),
			__RetSz
		);
	else if constexpr (__InpRnk == 3)
		return Functional::FromShared(
			Dimensions{ __Shape.c, __Shape.h, __Shape.w },
			// ReSharper disable once CppLambdaCaptureNeverUsed
			std::shared_ptr<void>(__Data, [__Out](void*) {}),
			__RetSz
		);
	else if constexpr (__InpRnk == 4)
		return Functional::FromShared(
			Dimensions{ __Shape.c, __Shape.d, __Shape.h, __Shape.w },
			// ReSharper disable once CppLambdaCaptureNeverUsed
			std::shared_ptr<void>(__Data, [__Out](void*) {}),
			__RetSz
		);
}

_D_Dragonian_NCNN_Lib_Space_End