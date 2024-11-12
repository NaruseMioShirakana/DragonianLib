#pragma once
#include "BaseF0Extractor.hpp"
#include "PluginBase/PluginBase.h"

_D_Dragonian_Lib_F0_Extractor_Header

class PluginF0Extractor : public BaseF0Extractor
{
public:
	/**
	 * @brief GetF0Size function type (Instance, SamplingRate, HopSize, UserParameters)
	 */
	using GetF0SizeFunctionType = size_t(*)(void*, long, long, void*);

	/**
	 * @brief Extract function type (Instance, Input, InputSize, SamplingRate, HopSize, F0Bins, F0Max, F0Min, UserParameters, Output)
	 */
	using ExtractFunctionType = void(*)(void*, const void*, size_t, long, long, long, double, double, void*, float*);

	PluginF0Extractor(const Plugin::Plugin& Plugin, const void* UserParameter);
	~PluginF0Extractor() override;

	Vector<float> ExtractF0(
		const Vector<double>& PCMData,
		const F0ExtractorParams& Params
	) override;

	Vector<float> ExtractF0(
		const Vector<float>& PCMData,
		const F0ExtractorParams& Params
	) override;

	Vector<float> ExtractF0(
		const Vector<int16_t>& PCMData,
		const F0ExtractorParams& Params
	) override;

private:
	void* _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;
	GetF0SizeFunctionType _MyGetF0Size = nullptr; ///< "void GetF0Size(void*, long, long, void*)"
	ExtractFunctionType _MyExtractPD = nullptr; ///< "void ExtractF0PD(void*, const void*, size_t, long, long, long, double, double, void*, float*)"
	ExtractFunctionType _MyExtractPS = nullptr; ///< "void ExtractF0PS(void*, const void*, size_t, long, long, long, double, double, void*, float*)"
	ExtractFunctionType _MyExtractI16 = nullptr; ///< "void ExtractF0I16(void*, const void*, size_t, long, long, long, double, double, void*, float*)"

	PluginF0Extractor(const PluginF0Extractor&) = delete;
	PluginF0Extractor& operator=(const PluginF0Extractor&) = delete;
	PluginF0Extractor(PluginF0Extractor&&) = delete;
	PluginF0Extractor& operator=(PluginF0Extractor&&) = delete;
};

_D_Dragonian_Lib_F0_Extractor_End