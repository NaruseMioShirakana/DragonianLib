/**
 * @file KmeansCluster.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief K-means cluster of DragonianLib
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Cluster/BaseCluster.hpp"
#include "Libraries/Cluster/K-DimensionalTree/KDTree.hpp"
#include <string>

_D_Dragonian_Lib_Cluster_Namespace_Begin

/**
 * @class KMeansCluster
 * @brief K-means cluster of DragonianLib
 */
class KMeansCluster : public BaseCluster
{
public:
	KMeansCluster() = delete;
	~KMeansCluster() override = default;

	/**
	 * @brief Construct a new KMeansCluster object
	 * @param ClusterRootPath The root directory of the cluster file, the files must be a numpy file with the name "KMeans.npy", and the shape of the file must be [CodebookSize, ClusterSize, Dimension]
	 * @param Dimension Dimension of the Unit feature
	 * @param ClusterSize Count of the cluster centers
	 */
	KMeansCluster(const std::wstring& ClusterRootPath, Int64 Dimension, Int64 ClusterSize);

	/**
	 * @brief Search for the nearest Points to the Point in the cluster of id, not need to call evaluate function of the returned tensor
	 * @param Points Point that needs to be searched, Shape: [Count, Dimension]
	 * @param CodebookID Codebook ID
	 * @param Count Count of the input and output Points
	 * @return A Tensor of the nearest Points, Shape: [Count, Dimension]
	 */
	Tensor<Float32, 2, Device::CPU> Search(Float32* Points, Long CodebookID, Int64 Count = 1) override;
protected:
	std::vector<KDTree::KDTree> _MyTrees;
private:
	KMeansCluster(const KMeansCluster&) = delete;
	KMeansCluster(KMeansCluster&&) = delete;
	KMeansCluster operator=(const KMeansCluster&) = delete;
	KMeansCluster operator=(KMeansCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End