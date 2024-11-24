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
#include "Libraries/K-DimensionalTree/KDTree.hpp"
#include <string>

_D_Dragonian_Lib_Cluster_Namespace_Begin

class KMeansCluster : public BaseCluster
{
public:
	KMeansCluster() = delete;
	~KMeansCluster() override = default;
	
	KMeansCluster(const std::wstring& _RootPath, size_t Dims, size_t KMeansLen);
	DragonianLibSTL::Vector<float> Search(float* Point, long SpeakerId, int64_t Count = 1) override;
protected:
	std::vector<KDTree::KDTree> _tree;
	size_t dims = 0;
private:
	KMeansCluster(const KMeansCluster&) = delete;
	KMeansCluster(KMeansCluster&&) = delete;
	KMeansCluster operator=(const KMeansCluster&) = delete;
	KMeansCluster operator=(KMeansCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End