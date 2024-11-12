/**
 * FileName: DioF0Extractor.hpp
 * Note: DragonianLib Dio F0Extractor
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib library.
 * DragonianLib library is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include "BaseF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

class DioF0Extractor : public BaseF0Extractor
{
public:
	DioF0Extractor() = default;
	~DioF0Extractor() override = default;

	static Vector<double> Dio(
		const Vector<double>& PCMData,
		const F0ExtractorParams& Params
	);

	Vector<float> ExtractF0(
		const Vector<double>& PCMData,
		const F0ExtractorParams& Params
	) override;
private:
	DioF0Extractor(const DioF0Extractor&) = delete;
	DioF0Extractor(DioF0Extractor&&) = delete;
	DioF0Extractor operator=(const DioF0Extractor&) = delete;
	DioF0Extractor operator=(DioF0Extractor&&) = delete;
};

_D_Dragonian_Lib_F0_Extractor_End
