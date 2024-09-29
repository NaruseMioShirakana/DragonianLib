/**
 * FileName: ClusterManager.hpp
 * Note: DragonianLib 聚类管理
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

	using GetClusterFn = std::function<ClusterWrp(const std::wstring&, size_t, size_t)>;

	void RegisterCluster(const std::wstring& _name, const GetClusterFn& _constructor_fn);

	/**
	 * \brief 获取聚类
	 * \param _name 类名
	 * \param _path 聚类数据路径
	 * \param hidden_size hubert维数
	 * \param KmeansLen 聚类的长度
	 * \return 聚类
	 */
	ClusterWrp GetCluster(const std::wstring& _name, const std::wstring& _path, size_t hidden_size, size_t KmeansLen);

}