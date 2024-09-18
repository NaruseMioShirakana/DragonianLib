/**
 * FileName: BaseCluster.hpp
 * Note: DragonianLib 聚类基类
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

namespace DragonianLib
{
	class BaseCluster
	{
	public:
		BaseCluster() = default;
		virtual ~BaseCluster() = default;

		/**
		 * \brief 查找聚类最邻近点
		 * \param point 待查找的点
		 * \param sid 角色ID
		 * \param n_points 点数
		 * \return 查找到的最邻近点
		 */
		virtual DragonianLibSTL::Vector<float> Search(float* point, long sid, int64_t n_points = 1);
		BaseCluster(const BaseCluster&) = delete;
		BaseCluster(BaseCluster&&) = delete;
		BaseCluster operator=(const BaseCluster&) = delete;
		BaseCluster operator=(BaseCluster&&) = delete;
	};
}

