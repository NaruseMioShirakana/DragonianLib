/**
 * FileName: TensorExtractorManager.hpp
 * Note: MoeVoiceStudioCore TensorExtractorManager
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include <functional>
#include "BaseTensorExtractor.hpp"

LibSvcHeader

using TensorExtractor = std::shared_ptr<LibSvcTensorExtractor>;

using GetTensorExtractorFn = std::function<TensorExtractor(uint64_t, uint64_t, uint64_t, bool, bool, uint64_t, uint64_t, const LibSvcTensorExtractor::Others&)>;

/**
 * @brief Register a tensor extractor
 * @param _name Name of the tensor extractor
 * @param _constructor_fn Constructor function of the tensor extractor
 */
void RegisterTensorExtractor(
	const std::wstring& _name,
	const GetTensorExtractorFn& _constructor_fn
);

/**
 * @brief Get a tensor extractor
 * @param _name Name of the tensor extractor
 * @param _srcsr Source sample rate
 * @param _sr Sample rate
 * @param _hop Hop size
 * @param _smix Use speaker mix
 * @param _volume Use volume
 * @param _hidden_size Hidden size
 * @param _nspeaker Number of speakers
 * @param _other Other parameters
 * @return Tensor extractor
 */
TensorExtractor GetTensorExtractor(
	const std::wstring& _name,
	uint64_t _srcsr, uint64_t _sr, uint64_t _hop,
	bool _smix, bool _volume, uint64_t _hidden_size,
	uint64_t _nspeaker,
	const LibSvcTensorExtractor::Others& _other
);

LibSvcEnd