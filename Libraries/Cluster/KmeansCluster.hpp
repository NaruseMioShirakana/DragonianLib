/**
 * FileName: KmeansCluster.hpp
 * Note: DragonianLib Kmeans cluster
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
#include "BaseCluster.hpp"
#include "K-DimensionalTree/KDTree.hpp"
#include <string>

namespace DragonianLib {

	class KMeansCluster : public BaseCluster
	{
	public:
		KMeansCluster() = delete;
		~KMeansCluster() override = default;
		KMeansCluster(const KMeansCluster&) = delete;
		KMeansCluster(KMeansCluster&&) = delete;
		KMeansCluster operator=(const KMeansCluster&) = delete;
		KMeansCluster operator=(KMeansCluster&&) = delete;
		KMeansCluster(const std::wstring& _path, size_t hidden_size, size_t KmeansLen);
		DragonianLibSTL::Vector<float> Search(float* point, long sid, int64_t n_points = 1) override;
	private:
		std::vector<KDTree> _tree;
		size_t dims = 0;
	};

}