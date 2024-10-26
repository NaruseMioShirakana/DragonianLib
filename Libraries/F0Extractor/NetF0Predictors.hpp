/**
 * FileName: NetF0Predictors.hpp
 * Note: DragonianLib RMVPE & FCPE F0 Extractor
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

#ifdef DRAGONIANLIB_ONNXRT_LIB

#include "BaseF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

class RMVPEF0Extractor : public BaseF0Extractor
{
public:
	RMVPEF0Extractor(int sampling_rate, int hop_size, int n_f0_bins = 256, double max_f0 = 1100.0, double min_f0 = 50.0);
	~RMVPEF0Extractor() override = default;
	RMVPEF0Extractor(const RMVPEF0Extractor&) = delete;
	RMVPEF0Extractor(RMVPEF0Extractor&&) = delete;
	RMVPEF0Extractor operator=(const RMVPEF0Extractor&) = delete;
	RMVPEF0Extractor operator=(RMVPEF0Extractor&&) = delete;

	//void InterPf0(size_t TargetLength);
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength) override;
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<float>& PCMData, size_t TargetLength) override;
private:
	DragonianLibSTL::Vector<const char*> InputNames = { "waveform", "threshold" };
	DragonianLibSTL::Vector<const char*> OutputNames = { "f0", "uv" };
	DragonianLibSTL::Vector<double> refined_f0;
};

class MELPEF0Extractor : public BaseF0Extractor
{
public:
	MELPEF0Extractor(int sampling_rate, int hop_size, int n_f0_bins = 256, double max_f0 = 1100.0, double min_f0 = 50.0);
	~MELPEF0Extractor() override = default;
	MELPEF0Extractor(const MELPEF0Extractor&) = delete;
	MELPEF0Extractor(MELPEF0Extractor&&) = delete;
	MELPEF0Extractor operator=(const MELPEF0Extractor&) = delete;
	MELPEF0Extractor operator=(MELPEF0Extractor&&) = delete;

	//void InterPf0(size_t TargetLength);
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength) override;
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<float>& PCMData, size_t TargetLength) override;
private:
	DragonianLibSTL::Vector<const char*> InputNames = { "waveform" };
	DragonianLibSTL::Vector<const char*> OutputNames = { "f0" };
	DragonianLibSTL::Vector<double> refined_f0;
};

/**
 * @brief Load FCPE Model
 * @param FCPEModelPath Path to FCPE Model
 * @param Env Environment of ONNX Runtime
 */
void LoadFCPEModel(const wchar_t* FCPEModelPath, const DragonianLibOrtEnv& Env);

/**
 * @brief Load RMVPE Model
 * @param RMVPEModelPath Path to RMVPE Model
 * @param Env Environment of ONNX Runtime
 */
void LoadRMVPEModel(const wchar_t* RMVPEModelPath, const DragonianLibOrtEnv& Env);

/**
 * @brief Unload FCPE Model
 */
void UnloadFCPEModel();

/**
 * @brief Unload RMVPE Model
 */
void UnloadRMVPEModel();

_D_Dragonian_Lib_F0_Extractor_End

#endif
