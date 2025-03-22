/**
 * @file ClusterManager.hpp
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
 * @brief Cluster manager for DragonianLib
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "BaseCluster.hpp"

_D_Dragonian_Lib_Cluster_Namespace_Begin

using Cluster = std::shared_ptr<BaseCluster>;
using Constructor = std::function<Cluster(const std::wstring&, Int64, Int64)>;

/**
 * @brief Register All Clusters in the directory
 * @param _PluginRootDirectory Root directory of the Clusters
 */
void RegisterClusters(
	const std::wstring& _PluginRootDirectory
);

/**
 * @brief Register a Cluster
 * @param _PluginName Name of the Cluster
 * @param _Constructor Constructor of the Cluster
 */
void RegisterCluster(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
);

/**
 * @brief Create a new Cluster instance
 * @param ClusterName Name of the Cluster
 * @param ClusterFile File of the Cluster
 * @param ClusterDimension Dimension of the Cluster
 * @param ClusterSize Size of the Cluster
 * @return Cluster
 */
Cluster New(
	const std::wstring& ClusterName,
	const std::wstring& ClusterFile,
	Int64 ClusterDimension,
	Int64 ClusterSize
);

/**
 * @brief Get a list of Cluster names
 * @return List of Cluster names
 */
const std::vector<std::wstring>& GetClusterList();

_D_Dragonian_Lib_Cluster_Namespace_End