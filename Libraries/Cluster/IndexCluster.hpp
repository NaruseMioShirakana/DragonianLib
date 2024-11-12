/**
 * FileName: IndexCluster.hpp
 * Note: DragonianLib Index Cluster
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
#include <memory>
#include <string>
#include <vector>
#include "BaseCluster.hpp"

_D_Dragonian_Lib_Cluster_Namespace_Begin

class IndexClusterCore;

class IndexCluster : public BaseCluster
{
public:
    IndexCluster() = delete;
    ~IndexCluster() override = default;
    
    IndexCluster(const std::wstring& RootPath, size_t, size_t);
    DragonianLibSTL::Vector<float> Search(float* Point, long SpeakerID, int64_t PointCount = 1) override;
protected:
    std::vector<std::shared_ptr<IndexClusterCore>> Indexs;
    size_t n_hidden_size = 256;
private:
    IndexCluster(const IndexCluster&) = delete;
    IndexCluster(IndexCluster&&) = delete;
    IndexCluster operator=(const IndexCluster&) = delete;
    IndexCluster operator=(IndexCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End
