#pragma once
#include "base.h"

namespace DragonianLib
{
	namespace Util
	{
		using Byte = unsigned char;

		struct NumpyHeader
		{
			Byte magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
			Byte majorVersion = 1;
			Byte minorVersion = 0;
			uint16_t headerLength = 118;
		};

		size_t GetNumpyTypeAligsize(const std::string& _Type);

		std::pair<std::vector<int64_t>, std::vector<Byte>> LoadNumpyFile(const std::wstring& _Path);

		CudaModules::Module::DictType LoadNumpyFileToDict(
			const std::wstring& _Path
		);
	}
}
