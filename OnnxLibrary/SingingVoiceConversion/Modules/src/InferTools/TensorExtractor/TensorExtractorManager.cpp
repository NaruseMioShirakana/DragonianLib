#include "../../../header/InferTools/TensorExtractor/TensorExtractorManager.hpp"
#include <map>
#include "Base.h"
#include "Util/Logger.h"

LibSvcHeader

std::map<std::wstring, GetTensorExtractorFn> RegisteredTensorExtractors;

void RegisterTensorExtractor(const std::wstring& _name, const GetTensorExtractorFn& _constructor_fn)
{
	if (RegisteredTensorExtractors.contains(_name))
	{
		DragonianLibLogMessage(L"[Warn] TensorExtractorNameConflict");
		return;
	}
	RegisteredTensorExtractors[_name] = _constructor_fn;
}

TensorExtractor GetTensorExtractor(const std::wstring& _name, uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const LibSvcTensorExtractor::Others& _other)
{
	const auto f_TensorExtractor = RegisteredTensorExtractors.find(_name);
	if (f_TensorExtractor != RegisteredTensorExtractors.end())
		return f_TensorExtractor->second(_srcsr, _sr, _hop, _smix, _volume, _hidden_size, _nspeaker, _other);
	throw std::runtime_error("Unable To Find An Available TensorExtractor");
}

LibSvcEnd