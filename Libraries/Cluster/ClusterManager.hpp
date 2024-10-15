/**
 * FileName: ClusterManager.hpp
 * Note: DragonianLib ClusterManager Header File
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
#include <functional>
#include <memory>
#include <string>

namespace DragonianLib {
	
	using ClusterWrp = std::shared_ptr<BaseCluster>;

	//Cluster Constructor Function
	using GetClusterFn = std::function<ClusterWrp(const std::wstring&, size_t, size_t)>;

	/**
	 * @brief Register a constructor function for a cluster
	 * @param ClusterName Name of the cluster
	 * @param Constructor Constructor function
	 */
	void RegisterCluster(const std::wstring& ClusterName, const GetClusterFn& Constructor);

	/**
	 * @brief Get a cluster by name
	 * @param ClusterName Name of the cluster
	 * @param ClusterFile File path of the cluster
	 * @param ClusterDimension Dimension of the cluster
	 * @param ClusterSize Size of the cluster
	 * @return A shared pointer to the cluster
	 */
	ClusterWrp GetCluster(const std::wstring& ClusterName, const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize);

}