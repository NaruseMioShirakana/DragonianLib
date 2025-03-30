#include "../RMVPE.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

void InitNetPE()
{
	RegisterF0Extractor(
		L"RMVPE",
		[](const void* _ModelHParams) -> std::shared_ptr<BaseF0Extractor>
		{
			return std::make_shared<RMVPE>(_ModelHParams);
		}
	);
	RegisterF0Extractor(
		L"rmvpe",
		[](const void* _ModelHParams) -> std::shared_ptr<BaseF0Extractor>
		{
			return std::make_shared<RMVPE>(_ModelHParams);
		}
	);
	RegisterF0Extractor(
		L"FCPE",
		[](const void* _ModelHParams) -> std::shared_ptr<BaseF0Extractor>
		{
			return std::make_shared<FCPE>(_ModelHParams);
		}
	);
	RegisterF0Extractor(
		L"fcpe",
		[](const void* _ModelHParams) -> std::shared_ptr<BaseF0Extractor>
		{
			return std::make_shared<FCPE>(_ModelHParams);
		}
	);
}

_D_Dragonian_Lib_F0_Extractor_End