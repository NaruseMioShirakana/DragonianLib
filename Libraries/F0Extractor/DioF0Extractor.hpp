﻿/**
 * FileName: DioF0Extractor.hpp
 * Note: DragonianLib 官方F0提取算法 Dio
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

DragonianLibF0ExtractorHeader
class DioF0Extractor : public BaseF0Extractor
{
public:
	DioF0Extractor(int sampling_rate, int hop_size, int n_f0_bins = 256, double max_f0 = 1100.0, double min_f0 = 50.0);
	~DioF0Extractor() override = default;
	DioF0Extractor(const DioF0Extractor&) = delete;
	DioF0Extractor(DioF0Extractor&&) = delete;
	DioF0Extractor operator=(const DioF0Extractor&) = delete;
	DioF0Extractor operator=(DioF0Extractor&&) = delete;

	void compute_f0(const double* PCMData, size_t PCMLen);
	void InterPf0(size_t TargetLength);
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength) override;
private:
	DragonianLibSTL::Vector<double> refined_f0;
};
DragonianLibF0ExtractorEnd