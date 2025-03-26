#include "../Samplers.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

const char* DenoiseFnInput[] = { "noise", "time", "condition" };
const char* DenoiseFnOutput[] = { "noise_pred" };
const char* PredFnInput[] = { "noise", "noise_pred", "time", "time_prev" };
const char* PredFnOutput[] = { "noise_pred_o" };
const char* AlphaFnInput[] = { "time" };
const char* AlphaFnOutput[] = { "alphas_cumprod" };

const char* VelocityFnInput[] = { "x", "t", "cond" };
const char* VelocityFnOutput[] = { "o" };

struct NoiseList
{
	Float32* Data1; Float32* Data2; Float32* Data3; Float32* Data4;
	NoiseList(UInt64 Size) : Data1(new Float32[Size]), Data2(new Float32[Size]), Data3(new Float32[Size]), Data4(new Float32[Size]) {}
	~NoiseList() { delete[] Data1; delete[] Data2; delete[] Data3; delete[] Data4; }
	NoiseList(const NoiseList&) = delete;
	NoiseList(NoiseList&&) = delete;
	NoiseList& operator=(const NoiseList&) = delete;
	NoiseList& operator=(NoiseList&&) = delete;
	Float32* operator[](Int64 Index) { return reinterpret_cast<Float32**>(this)[Index % 4]; }
};

Ort::Value SampleDDim(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const DiffusionParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _DenoiseFn,
	const OnnxRuntimeModel& _NoisePredictorFn,
	const OnnxRuntimeModel& _AlphaCumprodFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& _Logger
);

Ort::Value SamplePndm(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const DiffusionParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _DenoiseFn,
	const OnnxRuntimeModel& _NoisePredictorFn,
	const OnnxRuntimeModel& _AlphaCumprodFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& _Logger
)
{
	if (!_DenoiseFn)
		_D_Dragonian_Lib_Throw_Exception("Denoiser is required for sampler");
	if (!_NoisePredictorFn && !_AlphaCumprodFn)
		_D_Dragonian_Lib_Throw_Exception("NoisePredictor/AlphaCumprod is required for sampler");
	if (!_NoisePredictorFn && _AlphaCumprodFn)
	{
		if (_Logger)
			_Logger->LogError(L"NoisePredictor is required for Pndm sampler, but AlphaCumprod is provided, using DDim sampler instead");
		return SampleDDim(
		   std::move(_Mel),
		   std::move(_Condition),
		   _Params,
		   _RunOptions,
		   _MemoryInfo,
		   _DenoiseFn,
		   _NoisePredictorFn,
		   _AlphaCumprodFn,
		   _ProgressCallbackFn,
			_Logger
	   );
	}

	if (_Params.Begin == _Params.End)
		_D_Dragonian_Lib_Throw_Exception("Invalid diffusion parameters, begin and end must not be equal");

	Int64 TArr[1], TPArr[1];
	constexpr Int64 TShape[1] = { 1 };

	const auto NoiseSize = _Mel.GetTensorTypeAndShapeInfo().GetElementCount();
	NoiseList Noises(NoiseSize);

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; } DenoiseIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition)
	};

	struct { Ort::Value X; Ort::Value XP; Ort::Value T; Ort::Value TP; } PredIn{
		Ort::Value{ nullptr },
		Ort::Value{ nullptr },
		Ort::Value{ nullptr },
		Ort::Value::CreateTensor(_MemoryInfo, TPArr, 1, TShape, 1)
	};

	Int64 Progress = 0;
	const auto TotalDiffusionSteps = std::max((_Params.End - _Params.Begin) / _Params.Stride, 1ll);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalDiffusionSteps);

	auto DenoiseFn = _DenoiseFn.Get();
	auto NoisePredictorFn = _NoisePredictorFn.Get();

	for (auto t = _Params.End; t > _Params.Begin; t -= _Params.Stride)
	{
		DenoiseIn.T.GetTensorMutableData<Int64>()[0] = t;
		PredIn.TP.GetTensorMutableData<Int64>()[0] = t - _Params.Stride > 0 ? t - _Params.Stride : 0;

		if (Progress == 0)
		{
			try
			{
				//noise_pred = denoise(x, t, condition)
				Ort::Value NoisePred{ nullptr };
				DenoiseFn->Run(
					_RunOptions,
					DenoiseFnInput,
					reinterpret_cast<const Ort::Value*>(&DenoiseIn),
					3,
					DenoiseFnOutput,
					&NoisePred,
					1
				);

				memcpy(
					Noises[Progress],
					NoisePred.GetTensorData<Float32>(),
					NoisePred.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(Float32)
				);
				PredIn.X = std::move(DenoiseIn.X); //x
				PredIn.XP = std::move(NoisePred);  //noise_pred
				PredIn.T = std::move(DenoiseIn.T); //t
				//PredIn.TP = std::move(PredIn.TP) //t_prev
			}
			catch (Ort::Exception& e1)
			{
				_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Denoiser\n") + e1.what());
			}
			try
			{
				//x_pred = get_x_pred(x, noise_pred, t, t_prev)
				Ort::Value XPred{ nullptr };
				NoisePredictorFn->Run(
					_RunOptions,
					PredFnInput,
					reinterpret_cast<const Ort::Value*>(&PredIn),
					4,
					PredFnOutput,
					&XPred,
					1
				);

				DenoiseIn.X = std::move(XPred);					//x_pred
				DenoiseIn.T = std::move(PredIn.TP);				//t_prev
				//DenoiseIn.Cond = std::move(DenoiseIn.Cond);	//condition
			}
			catch (Ort::Exception& e1)
			{
				_D_Dragonian_Lib_Throw_Exception(std::string("Locate: NoisePredictor\n") + e1.what());
			}

			try
			{
				//noise_pred_prev = denoise(x_pred, t_prev, condition)
				Ort::Value NoisePredPrev{ nullptr };
				DenoiseFn->Run(
					_RunOptions,
					DenoiseFnInput,
					reinterpret_cast<const Ort::Value*>(&DenoiseIn),
					3,
					DenoiseFnOutput,
					&NoisePredPrev,
					1
				);

				//noise_pred_prime = (noise_pred + noise_pred_prev) / 2
				const Float32* NoisePredData = Noises[Progress];
				auto NoisePredPrevData = NoisePredPrev.GetTensorMutableData<Float32>();
				const Float32* const NoisePredPrevDataEnd = NoisePredPrevData + NoiseSize;
				while (NoisePredPrevData < NoisePredPrevDataEnd)
					(*(NoisePredPrevData++) += *(NoisePredData++)) /= 2.0f;

				//PredIn.X = std::move(PredIn.X);		//x
				PredIn.XP = std::move(NoisePredPrev);	//noise_pred_prime
				//PredIn.T = std::move(PredIn.T);		//t
				PredIn.TP = std::move(DenoiseIn.T);		//t_prev
			}
			catch (Ort::Exception& e1)
			{
				_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Denoiser\n") + e1.what());
			}
		}
		else
		{
			//noise_pred = denoise(x, t, condition)
			Ort::Value NoisePred{ nullptr };
			try
			{
				DenoiseFn->Run(
					_RunOptions,
					DenoiseFnInput,
					reinterpret_cast<const Ort::Value*>(&DenoiseIn),
					3,
					DenoiseFnOutput,
					&NoisePred,
					1
				);
			}
			catch (Ort::Exception& e1)
			{
				_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Denoiser\n") + e1.what());
			}

			memcpy(
				Noises[Progress],
				NoisePred.GetTensorData<Float32>(),
				NoisePred.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(Float32)
			);

			auto NoisePredPrimeData = NoisePred.GetTensorMutableData<Float32>();
			if (Progress == 1)
				for (size_t it = 0; it < NoiseSize; ++it)
					((*(NoisePredPrimeData++) *= 3.0f) -= Noises[Progress - 1][it]) /= 2.0f;
			else if (Progress == 2)
				for (size_t it = 0; it < NoiseSize; ++it)
					(((*(NoisePredPrimeData++) *= 23.0f) -= Noises[Progress - 1][it] * 16.0f) += Noises[Progress - 2][it] * 5.0f) /= 12.0f;
			else
				for (size_t it = 0; it < NoiseSize; ++it)
					((((*(NoisePredPrimeData++) *= 55.0f) -= Noises[Progress - 1][it] * 59.0f) += Noises[Progress - 2][it] * 37.0f) -= Noises[Progress - 3][it] * 9.0f) /= 24.0f;

			PredIn.X = std::move(DenoiseIn.X);					//x
			PredIn.XP = std::move(NoisePred);					//noise_pred
			PredIn.T = std::move(DenoiseIn.T);					//t
			//PredIn.TP = std::move(PredIn.TP)					//t_prev
		}
		try
		{
			//x_prev = get_x_pred(x, noise_pred_prime, t, t_prev)
			Ort::Value XPrev{ nullptr };
			_NoisePredictorFn->Run(
				_RunOptions,
				PredFnInput,
				reinterpret_cast<const Ort::Value*>(&PredIn),
				4,
				PredFnOutput,
				&XPrev,
				1
			);
			DenoiseIn.X = std::move(XPrev);					//x_prev
			DenoiseIn.T = std::move(PredIn.T);				//t
			//DenoiseIn.Cond = std::move(DenoiseIn.Cond);	//condition
		}
		catch (Ort::Exception& e1)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: NoisePredictor\n") + e1.what());
		}
		++Progress;
		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Progress);
	}

	return std::move(DenoiseIn.X);
}

Ort::Value SampleDDim(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const DiffusionParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _DenoiseFn,
	const OnnxRuntimeModel& _NoisePredictorFn,
	const OnnxRuntimeModel& _AlphaCumprodFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& _Logger
)
{
	if (!_DenoiseFn)
		_D_Dragonian_Lib_Throw_Exception("Denoiser is required for sampler");
	if (!_NoisePredictorFn && !_AlphaCumprodFn)
		_D_Dragonian_Lib_Throw_Exception("NoisePredictor/AlphaCumprod is required for sampler");
	if (_NoisePredictorFn && !_AlphaCumprodFn)
	{
		if (_Logger)
			_Logger->LogError(L"AlphaCumprod is required for DDim sampler, but NoisePredictor is provided, using Pndm sampler instead");
		return SamplePndm(
			std::move(_Mel),
			std::move(_Condition),
			_Params,
			_RunOptions,
			_MemoryInfo,
			_DenoiseFn,
			_NoisePredictorFn,
			_AlphaCumprodFn,
			_ProgressCallbackFn,
			_Logger
		);
	}

	if (_Params.Begin == _Params.End)
		_D_Dragonian_Lib_Throw_Exception("Invalid diffusion parameters, begin and end must not be equal");

	Int64 TArr[1], TPArr[1];
	constexpr Int64 TShape[1] = { 1 };

	const auto NoiseSize = _Mel.GetTensorTypeAndShapeInfo().GetElementCount();

	Int64 Progress = 0;
	const auto TotalDiffusionSteps = std::max((_Params.End - _Params.Begin) / _Params.Stride, 1ll);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalDiffusionSteps);

	auto DenoiseFn = _DenoiseFn.Get();
	auto AlphaCumprod = _AlphaCumprodFn.Get();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value TP; } DenoiseIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor(_MemoryInfo, TPArr, 1, TShape, 1)
	};

	for (auto t = _Params.End; t > _Params.Begin; t -= _Params.Stride)
	{
		Float32 AT, APREV;
		DenoiseIn.T.GetTensorMutableData<Int64>()[0] = t;
		DenoiseIn.TP.GetTensorMutableData<Int64>()[0] = t - _Params.Stride > 0 ? t - _Params.Stride : 0;
		try
		{
			Ort::Value Alpha{ nullptr };
			AlphaCumprod->Run(
				Ort::RunOptions{ nullptr },
				AlphaFnInput,
				& DenoiseIn.T,
				1,
				AlphaFnOutput,
				& Alpha,
				1
			);
			AT = Alpha.GetTensorData<Float32>()[0];
			AlphaCumprod->Run(
				Ort::RunOptions{ nullptr },
				AlphaFnInput,
				& DenoiseIn.TP,
				1,
				AlphaFnOutput,
				& Alpha,
				1
			);
			APREV = Alpha.GetTensorData<Float32>()[0];
		}
		catch (Ort::Exception& e1)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: AlphaCumprod\n") + e1.what());
		}
		Ort::Value DenoiseOut{ nullptr };
		try
		{
			DenoiseFn->Run(
				_RunOptions,
				DenoiseFnInput,
				reinterpret_cast<const Ort::Value*>(&DenoiseIn),
				3,
				DenoiseFnOutput,
				&DenoiseOut,
				1
			);
		}
		catch (Ort::Exception& e1)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Denoiser\n") + e1.what());
		}

		const auto X = DenoiseIn.X.GetTensorData<float>();
		const auto NoisePred = DenoiseOut.GetTensorMutableData<float>();
		const auto SQAP = sqrt(APREV);
		const auto SQAT = sqrt(AT);
		const auto NPMOP = (sqrt((1 - APREV) / APREV) - sqrt((1 - AT) / AT));
		for (size_t i = 0; i < NoiseSize; ++i)
			NoisePred[i] = (X[i] / SQAT + NPMOP * NoisePred[i]) * SQAP;
		DenoiseIn.X = std::move(DenoiseOut);

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, ++Progress);
	}

	return std::move(DenoiseIn.X);
}

std::unordered_map<std::wstring, DiffusionSampler> DiffusionSamplers{
	{ L"DDim", SampleDDim },
	{ L"Pndm", SamplePndm },
	{ L"ddim", SampleDDim },
	{ L"pndm", SamplePndm }
};

void RegisterDiffusionSampler(const std::wstring& _Name, DiffusionSampler _Sampler)
{
	if (DiffusionSamplers.contains(_Name))
		_D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()->LogWarn(L"Diffusion sampler " + _Name + L" already exists, overwriting");
	DiffusionSamplers[_Name] = _Sampler;
}

DiffusionSampler GetDiffusionSampler(const std::wstring& _Name)
{
	if (!DiffusionSamplers.contains(_Name))
		_D_Dragonian_Lib_Throw_Exception("Diffusion sampler " + WideStringToUTF8(_Name) + " not found");
	return DiffusionSamplers[_Name];
}

Ort::Value ReflowEularSampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);
		Ort::Value VelocityOut{ nullptr };
		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&VelocityOut,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		auto X1 = VelocityIn.X.GetTensorMutableData<float>();
		auto KD = VelocityOut.GetTensorData<float>();

		for (size_t i = 0; i < XSize; ++i)
			X1[i] += KD[i] * DT;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

Ort::Value ReflowHeunSampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };
	auto XShape = _Mel.GetTensorTypeAndShapeInfo().GetShape();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value XT; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor<Float32>(GetDefaultOrtAllocator(), XShape.data(), 4)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	const auto X1 = VelocityIn.X.GetTensorMutableData<float>();
	const auto XT = VelocityIn.XT.GetTensorMutableData<float>();

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		Ort::Value K1{ nullptr }, K2{ nullptr };

		//1000 * t
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K1,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		std::swap(VelocityIn.X, VelocityIn.XT);
		auto K1D = K1.GetTensorData<float>();

		//x + k_1 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + K1D[i] * DT;
		//1000 * (t + dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K2,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		auto K2D = K2.GetTensorData<float>();

		//x += (k_1 + k_2) / 2 * dt
		for (size_t i = 0; i < XSize; ++i)
			X1[i] += (K1D[i] + K2D[i]) * 0.5f * DT;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		std::swap(VelocityIn.X, VelocityIn.XT);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

Ort::Value ReflowRk2Sampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };
	auto XShape = _Mel.GetTensorTypeAndShapeInfo().GetShape();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value XT; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor<Float32>(GetDefaultOrtAllocator(), XShape.data(), 4)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	const auto X1 = VelocityIn.X.GetTensorMutableData<float>();
	const auto XT = VelocityIn.XT.GetTensorMutableData<float>();

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		Ort::Value K1{ nullptr }, K2{ nullptr };

		//1000 * t
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K1,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		std::swap(VelocityIn.X, VelocityIn.XT);
		auto K1D = K1.GetTensorData<float>();

		//x + 0.5 * k_1 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.5f * K1D[i] * DT;
		//1000 * (t + 0.5 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.5f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K2,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		auto K2D = K2.GetTensorData<float>();

		//x += k_2 * dt
		for (size_t i = 0; i < XSize; ++i)
			X1[i] += K2D[i] * DT;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		std::swap(VelocityIn.X, VelocityIn.XT);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

Ort::Value ReflowRk4Sampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };
	auto XShape = _Mel.GetTensorTypeAndShapeInfo().GetShape();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value XT; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor<Float32>(GetDefaultOrtAllocator(), XShape.data(), 4)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	const auto X1 = VelocityIn.X.GetTensorMutableData<float>();
	const auto XT = VelocityIn.XT.GetTensorMutableData<float>();

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		Ort::Value K1{ nullptr }, K2{ nullptr }, K3{ nullptr }, K4{ nullptr };

		//1000 * t
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);
		
		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K1,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		std::swap(VelocityIn.X, VelocityIn.XT);
		const auto K1D = K1.GetTensorData<float>();

		//x + 0.5 * k_1 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.5f * K1D[i] * DT;
		//1000 * (t + 0.5 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.5f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K2,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K2D = K2.GetTensorData<float>();

		//x + 0.5 * k_2 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.5f * K2D[i] * DT;
		//1000 * (t + 0.5 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.5f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K3,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K3D = K3.GetTensorData<float>();

		//x + k_3 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + K3D[i] * DT;
		//1000 * (t + dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K4,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K4D = K4.GetTensorData<float>();

		//x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6 * dt
		for (size_t i = 0; i < XSize; ++i)
			X1[i] += (K1D[i] + 2.0f * K2D[i] + 2.0f * K3D[i] + K4D[i]) / 6.0f * DT;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		std::swap(VelocityIn.X, VelocityIn.XT);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

Ort::Value ReflowRk6Sampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };
	auto XShape = _Mel.GetTensorTypeAndShapeInfo().GetShape();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value XT; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor<Float32>(GetDefaultOrtAllocator(), XShape.data(), 4)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	const auto X1 = VelocityIn.X.GetTensorMutableData<float>();
	const auto XT = VelocityIn.XT.GetTensorMutableData<float>();

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		Ort::Value K1{ nullptr }, K2{ nullptr }, K3{ nullptr }, K4{ nullptr }, K5{ nullptr }, K6{ nullptr };

		//1000 * t
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K1,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		std::swap(VelocityIn.X, VelocityIn.XT);
		const auto K1D = K1.GetTensorData<float>();

		//x + 0.25 * k_1 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.25f * K1D[i] * DT;
		//1000 * (t + 0.25 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.25f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K2,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K2D = K2.GetTensorData<float>();

		//x + 0.125 * (k_2 + k_1) * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.125f * (K1D[i] + K2D[i]) * DT;
		//1000 * (t + 0.25 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.25f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K3,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K3D = K3.GetTensorData<float>();

		//x + 0.5 * (-k_2 + 2 * k_3) * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.5f * (-K2D[i] + 2.0f * K3D[i]) * DT;
		//1000 * (t + 0.5 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.5f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K4,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K4D = K4.GetTensorData<float>();

		//x + 0.0625 * (3 * k_1 + 9 * k_4) * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + 0.0625f * (3.0f * K1D[i] + 9.0f * K4D[i]) * DT;
		//1000 * (t + 0.75 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 0.75f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K5,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K5D = K4.GetTensorData<float>();

		//x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + (-3.0f * K1D[i] + 2.0f * K2D[i] + 12.0f * K3D[i] - 12.0f * K4D[i] + 8.0f * K5D[i]) * DT / 7.0f;
		//1000 * (t + dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K6,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K6D = K4.GetTensorData<float>();

		//x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
		for (size_t i = 0; i < XSize; ++i)
			X1[i] += (7.0f * K1D[i] + 32.0f * K3D[i] + 12.0f * K4D[i] + 32.0f * K5D[i] + 7.0f * K6D[i]) * DT / 90.0f;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		std::swap(VelocityIn.X, VelocityIn.XT);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

Ort::Value ReflowPECECESampler(
	Ort::Value&& _Mel,
	Ort::Value&& _Condition,
	const ReflowParameters& _Params,
	const Ort::RunOptions& _RunOptions,
	const Ort::MemoryInfo& _MemoryInfo,
	const OnnxRuntimeModel& _VelocityFn,
	const std::optional<ProgressCallback>& _ProgressCallbackFn,
	const DLogger& /*_Logger*/
)
{
	if (!_VelocityFn)
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for sampler");

	Int64 TArr[1];
	constexpr Int64 TShape[1] = { 1 };
	auto XShape = _Mel.GetTensorTypeAndShapeInfo().GetShape();

	struct { Ort::Value X; Ort::Value T; Ort::Value Cond; Ort::Value XT; } VelocityIn{
		std::move(_Mel),
		Ort::Value::CreateTensor(_MemoryInfo, TArr, 1, TShape, 1),
		std::move(_Condition),
		Ort::Value::CreateTensor<Float32>(GetDefaultOrtAllocator(), XShape.data(), 4)
	};

	auto VelocityFn = _VelocityFn.Get();
	const auto XSize = VelocityIn.X.GetTensorTypeAndShapeInfo().GetElementCount();

	const auto TotalSteps = static_cast<Int64>((_Params.End - _Params.Begin) / _Params.Stride);
	if (_ProgressCallbackFn)
		(*_ProgressCallbackFn)(true, TotalSteps);

	const auto X1 = VelocityIn.X.GetTensorMutableData<float>();
	const auto XT = VelocityIn.XT.GetTensorMutableData<float>();

	Float32 T = _Params.Begin;
	const Float32 DT = _Params.Stride;
	const Float32 Scale = _Params.Scale;
	for (auto Step : TemplateLibrary::Ranges(TotalSteps))
	{
		Ort::Value K1{ nullptr }, K2{ nullptr }, K3{ nullptr }, K4{ nullptr };

		//1000 * t
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t(T * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K1,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		std::swap(VelocityIn.X, VelocityIn.XT);
		const auto K1D = K1.GetTensorData<float>();

		//x + k_1 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + K1D[i] * DT;
		//1000 * (t + dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K2,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K2D = K2.GetTensorData<float>();

		//x + (k_1 + k_2) / 2 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] = X1[i] + (K1D[i] + K2D[i]) * 0.5f * DT;
		//1000 * (t + dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K3,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K3D = K3.GetTensorData<float>();

		//xt += k_3 * dt
		for (size_t i = 0; i < XSize; ++i)
			XT[i] += K3D[i] * DT;
		//1000 * (t + 2 * dt)
		*VelocityIn.T.GetTensorMutableData<int64_t>() = int64_t((T + 2.f * DT) * Scale);

		try
		{
			VelocityFn->Run(
				_RunOptions,
				VelocityFnInput,
				reinterpret_cast<const Ort::Value*>(&VelocityIn),
				3,
				VelocityFnOutput,
				&K4,
				1
			);
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(std::string("Locate: Velocity\n") + e.what());
		}
		const auto K4D = K4.GetTensorData<float>();

		//x += (k_3 + k_4) / 2 * dt
		for (size_t i = 0; i < XSize; ++i)
			X1[i] += (K3D[i] + K4D[i]) * 0.5f * DT;

		if (_ProgressCallbackFn)
			(*_ProgressCallbackFn)(false, Step + 1);
		std::swap(VelocityIn.X, VelocityIn.XT);
		T += DT;
	}
	return std::move(VelocityIn.X);
}

std::unordered_map<std::wstring, ReflowSampler> ReflowSamplers{
	{L"Eular", ReflowEularSampler},
	{L"Heun", ReflowHeunSampler},
	{L"Rk2", ReflowRk2Sampler},
	{L"Rk4", ReflowRk4Sampler},
	{L"Rk6", ReflowRk6Sampler},
	{L"PECECE", ReflowPECECESampler},
	{L"eular", ReflowEularSampler},
	{L"heun", ReflowHeunSampler},
	{L"rk2", ReflowRk2Sampler},
	{L"rk4", ReflowRk4Sampler},
	{L"rk6", ReflowRk6Sampler},
	{L"pecece", ReflowPECECESampler}
};

void RegisterReflowSampler(const std::wstring& _Name, ReflowSampler _Sampler)
{
	if (ReflowSamplers.contains(_Name))
		_D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()->LogWarn(L"Reflow sampler " + _Name + L" already exists, overwriting");
	ReflowSamplers[_Name] = _Sampler;
}

ReflowSampler GetReflowSampler(const std::wstring& _Name)
{
	if (!ReflowSamplers.contains(_Name))
		_D_Dragonian_Lib_Throw_Exception("Reflow sampler " + WideStringToUTF8(_Name) + " not found");
	return ReflowSamplers[_Name];
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End