#pragma once
/*

#pragma once

#ifdef DRAGONIANLIB_ONNXRT_LIB

#include "BaseF0Extractor.hpp"
#include "Libraries/EnvManager.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

class RMVPEF0Extractor : public BaseF0Extractor
{
public:
	RMVPEF0Extractor(const std::wstring& _ModelPath, const std::shared_ptr<DragonianLibOrtEnv>& _OrtEnv);
	~RMVPEF0Extractor() override = default;

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
	RMVPEF0Extractor(const RMVPEF0Extractor&) = delete;
	RMVPEF0Extractor(RMVPEF0Extractor&&) = delete;
	RMVPEF0Extractor operator=(const RMVPEF0Extractor&) = delete;
	RMVPEF0Extractor operator=(RMVPEF0Extractor&&) = delete;

	std::shared_ptr<Ort::Session> _MyModel = nullptr;
	std::shared_ptr<DragonianLibOrtEnv> _MyOrtEnv = nullptr;
	static inline Vector<const char*> InputNames = { "waveform", "threshold" };
	static inline Vector<const char*> OutputNames = { "f0", "uv" };
};

class MELPEF0Extractor : public BaseF0Extractor
{
public:
	MELPEF0Extractor(const std::wstring& _ModelPath, const std::shared_ptr<DragonianLibOrtEnv>& _OrtEnv);
	~MELPEF0Extractor() override = default;

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
	MELPEF0Extractor(const MELPEF0Extractor&) = delete;
	MELPEF0Extractor(MELPEF0Extractor&&) = delete;
	MELPEF0Extractor operator=(const MELPEF0Extractor&) = delete;
	MELPEF0Extractor operator=(MELPEF0Extractor&&) = delete;

	std::shared_ptr<Ort::Session> _MyModel = nullptr;
	std::shared_ptr<DragonianLibOrtEnv> _MyOrtEnv = nullptr;
	static inline Vector<const char*> InputNames = { "waveform" };
	static inline Vector<const char*> OutputNames = { "f0" };
};

_D_Dragonian_Lib_F0_Extractor_End

#endif
*/