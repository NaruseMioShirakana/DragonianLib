/**
 * @file IndexCluster.hpp
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
 * @brief Faiss index cluster of DragonianLib
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "BaseCluster.hpp"

_D_Dragonian_Lib_Cluster_Namespace_Begin

class IndexClusterCore;

/**
 * @class IndexCluster
 * @brief Faiss index cluster of DragonianLib
 */
class IndexCluster : public BaseCluster
{
public:
    IndexCluster() = delete;
    ~IndexCluster() override = default;

    /**
     * @brief Construct a new KMeansCluster object
	 * @param RootPath The root directory of the cluster file, the files must be a faiss file with the name "Index-[i].index", [i] means codebook id of this file
     * @param Dimension Dimension of the Unit feature
     */
    IndexCluster(const std::wstring& RootPath, Int64 Dimension, Int64);

    /**
     * @brief Search for the nearest Points to the Point in the cluster of id, not need to call evaluate function of the returned tensor
     * @param Point Point that needs to be searched, Shape: [Count, Dimension]
     * @param CodebookID Codebook ID
     * @param PointCount Count of the input and output Points
     * @return A Tensor of the nearest Points, Shape: [Count, Dimension]
     */
    Tensor<Float32, 2, Device::CPU> Search(Float32* Point, Long CodebookID, Int64 PointCount = 1) override;
protected:
    std::vector<std::shared_ptr<IndexClusterCore>> Indexs;
private:
    IndexCluster(const IndexCluster&) = delete;
    IndexCluster(IndexCluster&&) = delete;
    IndexCluster operator=(const IndexCluster&) = delete;
    IndexCluster operator=(IndexCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End
