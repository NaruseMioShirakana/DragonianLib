#include "../../../header/InferTools/TensorExtractor/TensorExtractorManager.hpp"
#include "../../../header/InferTools/TensorExtractor/TensorExtractor.hpp"
#include <map>
#include "Libraries/Base.h"
#include "Libraries/Util/Logger.h"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_Header

std::map<std::wstring, GetTensorExtractorFn> RegisteredTensorExtractors;

void RegisterTensorExtractor(const std::wstring& _name, const GetTensorExtractorFn& _constructor_fn)
{
	if (RegisteredTensorExtractors.contains(_name))
	{
		LogWarn(L"Name Of Tensor Extractor Already Exists");
		return;
	}
	RegisteredTensorExtractors[_name] = _constructor_fn;
}

TensorExtractor GetTensorExtractor(const std::wstring& _name, uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const LibSvcTensorExtractor::Others& _other)
{
	const auto f_TensorExtractor = RegisteredTensorExtractors.find(_name);
	if (f_TensorExtractor != RegisteredTensorExtractors.end())
		return f_TensorExtractor->second(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other);
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available Tensor Extractor");
}

struct Init
{
	Init()
	{
		RegisterTensorExtractor(L"SoVits2.0",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			-> TensorExtractor
			{
				return std::make_shared<SoVits2TensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"SoVits3.0",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			-> TensorExtractor
			{
				return std::make_shared<SoVits3TensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"SoVits4.0",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<SoVits4TensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"SoVits4.0-DDSP",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<SoVits4DDSPTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"RVC",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<RVCTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"DiffSvc",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<DiffSvcTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"DiffusionSvc",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<DiffusionSvcTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"ReflowSvc",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<DiffusionSvcTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"UnionSvc",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<DiffusionSvcTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});

		RegisterTensorExtractor(L"DDSPSvc",
			[](
				uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
				bool _smix, bool _volume, uint64_t _hidden_size,
				uint64_t _nspeaker,
				const LibSvcTensorExtractor::Others& _other
				)
			->TensorExtractor
			{
				return std::make_shared<DiffusionSvcTensorExtractor>(_srcsr, _sr, _hop, _smix, _volume,
					_hidden_size, _nspeaker, _other);
			});
	}
};
Init _Vardef_Init;

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_End