/**
 * @file PluginBasedCluster.hpp
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
 * @brief Cluster based on dynamic library plugins
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Cluster/BaseCluster.hpp"
#include "Libraries/PluginBase/PluginBase.h"

_D_Dragonian_Lib_Cluster_Namespace_Begin

struct PluginClusterInfo
{
	std::wstring ClusterFile;
	Int64 ClusterDimension;
	Int64 ClusterSize;
};

/**
 * @class PluginCluster
 * @brief Cluster based on dynamic library plugins
 */
class PluginCluster : public BaseCluster
{
public:
	using SearchFunctionType = void(*)(void*, const float*, long, int64_t, float*); ///< Search function type(Instance, Points, CodebookID, Count, OutputPtr)

	PluginCluster(const Plugin::Plugin& Plugin, const PluginClusterInfo& Params);
	~PluginCluster() override;

	Tensor<Float32, 2, Device::CPU> Search(Float32* Points, Long CodebookID, Int64 Count) override;

protected:
	void* _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;
	SearchFunctionType _MySearchFunction = nullptr; ///< "void ClusterSearch(void*, const float*, long, int64_t, float*)"
private:
	PluginCluster(const PluginCluster&) = delete;
	PluginCluster& operator=(const PluginCluster&) = delete;
	PluginCluster(PluginCluster&&) = delete;
	PluginCluster& operator=(PluginCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End