/**
 * @file BaseCluster.hpp
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
 * @brief Base class for clustering algorithms
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Tensor.h"

#define _D_Dragonian_Lib_Cluster_Namespace_Begin _D_Dragonian_Lib_Space_Begin namespace Cluster {
#define _D_Dragonian_Lib_Cluster_Namespace_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Cluster_Namespace_Begin

/**
 * @class BaseCluster
 * @brief Base class for clustering algorithms
 */
class BaseCluster
{
public:
	BaseCluster() = delete;
	BaseCluster(Int64 Dimension) : _MyDimension(Dimension) {}
	virtual ~BaseCluster() = default;

	/**
	 * @brief Search for the nearest Points to the Point in the cluster of id, not need to call evaluate function of the returned tensor
	 * @param Points Point that needs to be searched, Shape: [Count, Dimension]
	 * @param CodebookID Codebook ID
	 * @param Count Count of the input and output Points
	 * @return A Tensor of the nearest Points, Shape: [Count, Dimension]
	 */
	virtual Tensor<Float32, 2, Device::CPU> Search(Float32* Points, Long CodebookID, Int64 Count = 1);

	/**
	 * @brief Search for the nearest Points to the Point in the cluster of id, not need to call evaluate function of the returned tensor
	 * @param Points Point that needs to be searched, Shape: [Count, Dimension]
	 * @param CodebookID Codebook ID
	 * @return A Tensor of the nearest Points, Shape: [Count, Dimension]
	 */
	Tensor<Float32, 2, Device::CPU> Search(const Tensor<Float32, 2, Device::CPU>& Points, Long CodebookID);

protected:
	Int64 _MyDimension = 256;

private:
	BaseCluster(const BaseCluster&) = delete;
	BaseCluster(BaseCluster&&) = delete;
	BaseCluster operator=(const BaseCluster&) = delete;
	BaseCluster operator=(BaseCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End

