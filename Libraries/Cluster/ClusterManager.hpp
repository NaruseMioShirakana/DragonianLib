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
#include <memory>
#include <string>

_D_Dragonian_Lib_Cluster_Namespace_Begin

using Cluster = std::shared_ptr<BaseCluster>;

/**
 * @brief Register All Clusters in the directory
 * @param _PluginRootDirectory Root directory of the Clusters
 */
void RegisterCluster(const std::wstring& _PluginRootDirectory);

/**
 * @brief Get a Cluster
 * @param ClusterName Name of the Cluster
 * @param ClusterFile File of the Cluster
 * @param ClusterDimension Dimension of the Cluster
 * @param ClusterSize Size of the Cluster
 * @return Cluster
 */
Cluster GetCluster(
	const std::wstring& ClusterName,
	const std::wstring& ClusterFile,
	size_t ClusterDimension,
	size_t ClusterSize
);

/**
 * @brief Get a list of Cluster names
 * @return List of Cluster names
 */
const std::vector<std::wstring>& GetClusterList();

_D_Dragonian_Lib_Cluster_Namespace_End