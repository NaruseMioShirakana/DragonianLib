/**
 * FileName: BaseCluster.hpp
 * Note: DragonianLib Cluster Base
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
#include "MyTemplateLibrary/Vector.h"

#define _D_Dragonian_Lib_Cluster_Namespace_Begin namespace DragonianLib { namespace Cluster {
#define _D_Dragonian_Lib_Cluster_Namespace_End } }

_D_Dragonian_Lib_Cluster_Namespace_Begin

/**
 * @class BaseCluster
 * @brief Base class for clustering algorithms
 */
	class BaseCluster
{
public:
	BaseCluster() = default;
	virtual ~BaseCluster() = default;

	/**
	 * @brief Search for the nearest Points to the Point in the cluster of SpeakerId
	 * @param Point Point that needs to be searched
	 * @param SpeakerId SpeakerId of the cluster center vector
	 * @param Count Count of the nearest Points
	 * @return A vector of the nearest Points, Shape: [Count, Dimension]
	 */
	virtual DragonianLibSTL::Vector<float> Search(float* Point, long SpeakerId, int64_t Count = 1);

private:
	BaseCluster(const BaseCluster&) = delete;
	BaseCluster(BaseCluster&&) = delete;
	BaseCluster operator=(const BaseCluster&) = delete;
	BaseCluster operator=(BaseCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End

