/**
 * FileName: NativeApi.h
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
*/

#pragma once
#include "Libraries/DynLibExport.h"

#define _Dragonian_Lib_Svc_Add_Prefix(Name) DragonianVoiceSvc##Name
#define _Dragonian_Lib_Svc_Class_Name(Name) _Dragonian_Lib_Svc_Add_Prefix(Name##Class)
#define _Dragonian_Lib_Svc_Def_Api_Type(Name) \
	typedef struct _Dragonian_Lib_Svc_Class_Name(Name)* _Dragonian_Lib_Svc_Add_Prefix(Name);

#ifdef __GNUC__
#define _Dragonian_Lib_Svc_Deprecated __attribute__((deprecated))
#else
#ifdef _MSC_VER
#define _Dragonian_Lib_Svc_Deprecated __declspec(deprecated)
#endif
#endif
#ifdef _WIN32
#include "wtypes.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
	typedef signed char INT8, * PINT8;
	typedef signed short INT16, * PINT16;
	typedef signed int INT32, * PINT32;
	typedef signed long long INT64, * PINT64;
	typedef unsigned char UINT8, * PUINT8;
	typedef unsigned short UINT16, * PUINT16;
	typedef unsigned int UINT32, * PUINT32;
	typedef unsigned long long UINT64, * PUINT64;
	typedef wchar_t* NWPSTR, * LPWSTR, * PWSTR, * BSTR;
	typedef const wchar_t* LPCWSTR;
#endif

	enum _Dragonian_Lib_Svc_Add_Prefix(ModelType)
	{
		_Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV2),
			_Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV3),
			_Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV4),
			_Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV4b),
			_Dragonian_Lib_Svc_Add_Prefix(RVC),
			_Dragonian_Lib_Svc_Add_Prefix(DiffusionSvc),
			_Dragonian_Lib_Svc_Add_Prefix(ReflowSvc),
			_Dragonian_Lib_Svc_Add_Prefix(DDSPSvc),
	};

	/**
	 * @brief Key value pair, the key must be string and the value must be string
	 */
	typedef LPCWSTR _Dragonian_Lib_Svc_Add_Prefix(KVPair)[2];

	/**
	 * @brief Dictionary, array of key value pair, the last element of the array must be (nullptr, nullptr)
	 */
	typedef _Dragonian_Lib_Svc_Add_Prefix(KVPair)* _Dragonian_Lib_Svc_Add_Prefix(ArgDict);

	/**
	 * @brief Progress callback, it is the callback function for progress updates, if you need to get the progress of the inference, you must set this callback function, arguments of the callback function are (arg1, arg2), if arg1 is true, arg2 is the total steps of the inference, if arg1 is false, arg2 is the current step of the inference
	 */
	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(ProgressCallback))(bool, INT64);

	/**
	 * @brief Logger function
	 */
	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(LogFunction))(const wchar_t* Message, unsigned Level);

	/**
	 * @brief Deleter function
	 */
	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(Deleter))(void*);

	/**
	 * @brief DragonianLib API Enviroment, it is the environment of the DragonianLib API, it is used to set the execution provider and thread count
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(Enviroment);

	/**
	 * @brief Singing Voice Conversion Model, it is used to convert the singing voice to the target voice
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(Model);

	/**
	 * @brief Unit Encoder, it is used to encode the audio to the unit
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(UnitsEncoder);

	/**
	 * @brief Vocoder Model, it is used to convert the mel-spectrogram to the audio
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(Vocoder);

	/**
	 * @brief Cluster, it is used to search the nearest cluster center of units
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(Cluster);

	/**
	 * @brief F0 extractor, it is used to extract the F0 from the audio
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(F0Extractor);

	/**
	 * @brief FloatTensor, it is used to store the float tensor data
	 */
	_Dragonian_Lib_Svc_Def_Api_Type(FloatTensor);

#ifdef _MSC_VER
#pragma pack(push, 4)
#else
#pragma pack(4)
#endif

	/**
	 * @brief Enviroment settings
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting)
	{
		/**
		 * @brief Execution provider (device) of the environment, it is used to set the execution provider of the environment, the provider must be one of the following:
		 * - 0: CPU (default)
		 * - 1: CUDA (NVIDIA GPU)
		 * - 2: ROCm (AMD GPU), not implemented yet
		 * - 3: DIRECTX (Intel GPU)
		 */
		INT32 Provider;

		/**
		 * @brief Device ID of the environment, if the provider is CPU, this value will be ignored, if the provider is CUDA, this value will be ignored, you must set id in CUDAConfig with "device_id", if the provider is ROCm, this value will be the ROCm device ID, if the provider is DIRECTX, this value will be the DirectX device ID
		 */
		INT64 DeviceID;

		/**
		 * @brief Number of threads for intra-op parallelism.
		 */
		INT64 IntraOpNumThreads;

		/**
		 * @brief Number of threads for inter-op parallelism.
		 */
		INT64 InterOpNumThreads;

		/**
		 * @brief Log level of the environment, it is used to set the log level of the environment, the log level must be one of the following:
		 * - 0: ORT_LOGGING_LEVEL_VERBOSE
		 * - 1: ORT_LOGGING_LEVEL_INFO
		 * - 2: ORT_LOGGING_LEVEL_WARNING (default)
		 * - 3: ORT_LOGGING_LEVEL_ERROR
		 * - 4: ORT_LOGGING_LEVEL_FATAL
		 */
		INT32 LoggingLevel;

		/**
		 * @brief Logger id of the environment.
		 */
		LPCWSTR LoggerId;

		/**
		 * @brief CUDA config of the environment.
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ArgDict) CUDAConfig;
	};

	/**
	 * @brief Init the enviroment settings.
	 * @param _Input The enviroment settings.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitEnviromentSetting)(
		_Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting)* _Input
		);

	/**
	 * @brief Model parameters
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(HyperParameters)
	{
		/**
		 * @brief Model Type, it is the type of the model
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ModelType) ModelType;

		/**
		 * @brief Model paths, key value pairs of model type and model path. must end with [nullptr, nullptr]
		 *
		 * if the model is a diffusion model, the model type must be following:
		 * - "Ctrl": if your diffusion model has only one onnx model, this is the only path you need to provide
		 * - "Ctrl": the encoder layer of the diffusion model.
		 * - "Denoiser": the denoiser layer of the diffusion model.
		 * - "NoisePredictor": the noise predictor layer of the diffusion model.
		 * - "AlphaCumprod": the alpha cumprod layer of the diffusion model. [optional]
		 *
		 * if the model is a vits based model, the model type must be following:
		 * - "Model": the model path of the vits based model.
		 *
		 * if the model is a reflow model, the model type must be following:
		 * - "Ctrl": the encoder layer of the reflow model.
		 * - "Velocity": the velocity layer of the reflow model.
		 *
		 * if the model is a ddsp model, the model type must be following:
		 * - "Ctrl": the source model of the ddsp model.
		 * - "Velocity": the velocity model of the reflow model.
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ArgDict) ModelPaths;

		/**
		 * @brief Sampling rate of the output audio, it is the sampling rate of the model, not means the output audio will be resampled to this sampling rate, the output audio will be generated at this sampling rate
		 */
		INT64 OutputSamplingRate;

		/**
		 * @brief Units dimension, it is the dimension of the units, the units dimension must be greater than (0)
		 */
		INT64 UnitsDim;

		/**
		 * @brief Hop size, it is the hop size of the model, the hop size must be greater than (0)
		 */
		INT64 HopSize;

		/**
		 * @brief Speaker count, it is the count of the speaker, the speaker count must be greater than (0)
		 */
		INT64 SpeakerCount;

		/**
		 * @brief Has volume embedding, it is the flag of the volume embedding layer, if the model has volume embedding layer, this flag must be (true), otherwise, this flag must be (false)
		 */
		INT32 HasVolumeEmbedding;

		/**
		 * @brief Has speaker embedding, it is the flag of the speaker embedding layer, if the model has speaker embedding layer, this flag must be (true), otherwise, this flag must be (false)
		 */
		INT32 HasSpeakerEmbedding;

		/**
		 * @brief Has speaker mix layer, it is the flag of the speaker mix layer, if the model has speaker mix layer, this flag must be (true), otherwise, this flag must be (false)
		 */
		INT32 HasSpeakerMixLayer;

		/**
		 * @brief Spec max, it is the maximum value of the spectrogram, the spec max must be greater than (spec min)
		 */
		float SpecMax;

		/**
		 * @brief Spec min, it is the minimum value of the spectrogram, the spec min must be less than (spec max)
		 */
		float SpecMin;

		/**
		 * @brief F0 bin, it is the bin count of the f0, the f0 bin must be greater than (0)
		 */
		INT64 F0Bin;

		/**
		 * @brief F0 max, it is the maximum value of the f0, the f0 max must be greater than (f0 min)
		 */
		float F0Max;

		/**
		 * @brief F0 min, it is the minimum value of the f0, the f0 min must be less than (f0 max)
		 */
		float F0Min;

		/**
		 * @brief Mel bins, it is the bin count of the mel spectrogram, the mel bins must be greater than (0)
		 */
		INT64 MelBins;

		/**
		 * @brief Progress callback, it is the callback function for progress updates, if you need to get the progress of the inference, you must set this callback function, arguments of the callback function are (arg1, arg2), if arg1 is true, arg2 is the total steps of the inference, if arg1 is false, arg2 is the current step of the inference
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ProgressCallback) ProgressCallback;

		/**
		 * @brief Extented parameters, key value pairs of extended parameters, for example, the max step of the diffusion model("MaxStep": "1000")
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ArgDict) ExtendedParameters;
	};

	/**
	 * @brief Init the model parameters.
	 * @param _Input The model parameters.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitHyperParameters)(
		_Dragonian_Lib_Svc_Add_Prefix(HyperParameters)* _Input
		);

	/**
	 * @brief Inference parameters for diffusion
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(DiffusionParameters)
	{
		/**
		 * @brief Stride of the diffusion loop, every sample operation will skip (stride) steps, the diffusion step is ((end - begin) / stride), stride must be greater than (0) and less than (step)
		 */
		INT64 Stride;

		/**
		 * @brief Begining of the diffusion loop, sample operation will start from (begin) step, begin must be greater equal than (0) and less equal than (end)
		 */
		INT64 Begin;

		/**
		 * @brief End of the diffusion loop, sample operation will end at (end) step, end must be greater equal than (begin) and less equal than (max step)
		 */
		INT64 End;

		/**
		 * @brief Sampler of the diffusion, it is the sampling method of the diffusion loop, the sampler must be one of the following:
		 *  - "Pndm"			(default)
		 *	- "DDim"			(implemented)
		 *  - "Eular"			(not implemented)
		 *	- "RK4"				(not implemented)
		 *	- "DPM-Solver"		(not implemented)
		 *	- "DPM-Solver++"	(not implemented)
		 */
		LPCWSTR Sampler;

		/**
		 * @brief Mel factor, multiplied to the mel spectrogram, this argument is only used if the output audio has incorrect samples, this means that the mel spectrogram has incorrect unit, the mel factor is used to correct the mel spectrogram
		 */
		float MelFactor;

		/**
		 * @brief User parameters, this argument is used to pass user parameters to the diffusion model, the user parameters must be a pointer to the user parameters struct
		 */
		void* UserParameters;
	};

	/**
	 * @brief Inference parameters for reflow
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(ReflowParameters)
	{
		/**
		 * @brief Stride of the reflow loop, every sample operation will skip (stride) steps, the reflow step is ((end - begin) / stride), stride must be greater than (0) and less than (step)
		 */
		float Stride;

		/**
		 * @brief Begining of the reflow loop, sample operation will start from (begin) step, begin must be greater than (0) and less equal than (end)
		 */
		float Begin;

		/**
		 * @brief End of the reflow loop, sample operation will end at (end) step, end must be greater equal than (begin) and less equal than (max step)
		 */
		float End;

		/**
		 * @brief Scale of the reflow, it is the scale of the reflow loop, the scale must be greater than (0)
		 */
		float Scale;

		/**
		 * @brief Sampler of the reflow, it is the sampling method of the reflow loop, the sampler must be one of the following:
		 *  - "Eular"			(default)
		 *	- "RK4"				(implemented)
		 *	- "PECECE"			(implemented)
		 *	- "Heun"			(implemented)
		 */
		LPCWSTR Sampler;

		/**
		 * @brief Mel factor, multiplied to the mel spectrogram, this argument is only used if the output audio has incorrect samples, this means that the mel spectrogram has incorrect unit, the mel factor is used to correct the mel spectrogram
		 */
		float MelFactor;

		/**
		 * @brief User parameters, this argument is used to pass user parameters to the reflow model, the user parameters must be a pointer to the user parameters struct
		 */
		void* UserParameters;
	};

	/**
	 * @brief Inference parameters
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(InferenceParameters)
	{
		/**
		 * @brief Noise scale factor, multiplied to the noise, this argument may be harmful to the output audio or helpful to the output audio. noise scale has no range, but it is recommended to be in the range of (0, 1)
		 */
		float NoiseScale;

		/**
		 * @brief Speaker id, it is the index of the speaker embedding layer, if the model has speaker mixing layer, this argument is used to modify the speaker mixing tensor, if the model has no speaker mixing layer and has speaker embedding layer, this argument is used to select the speaker embedding feature, speaker id must be greater than (0) and less than (speaker count)
		 */
		INT64 SpeakerId;

		/**
		 * @brief Pitch offset, f0 will be multiplied by (2 ^ (offset / 12)), the pitch offset has no range, but it is recommended to be in the range of midi pitch (-128, 128)
		 */
		float PitchOffset;

		/**
		 * @brief Random seed, this argument is used to generate random numbers, has unknown effect on the output audio, it depends on your luck
		 */
		INT64 Seed;

		/**
		 * @brief Cluster rate, it is the rate of the cluster, the cluster rate must be greater than (0) and less than (1)
		 */
		float ClusterRate;

		/**
		 * @brief Whether the f0 has unvoice, if this flag is (true), F0 could have zero values, otherwise, zero values in F0 will be interpolated, this value MUST be set by user
		 */
		INT32 F0HasUnVoice;

		/**
		 * @brief Diffusion parameters
		 */
		_Dragonian_Lib_Svc_Add_Prefix(DiffusionParameters) Diffusion;

		/**
		 * @brief Reflow parameters
		 */
		_Dragonian_Lib_Svc_Add_Prefix(ReflowParameters) Reflow;

		/**
		 * @brief STFT noise scale for SoVitsSvc4.0-Beta, in general, this argument is not used
		 */
		float StftNoiseScale;

		/**
		 * @brief User parameters of f0 preprocess method, this argument is used to pass user parameters to the model, the user parameters must be a pointer to the user parameters struct, if the model not input the source f0(HZ) and the f0 preprocess method has user parameters, this argument must be set, otherwise, this argument must be (nullptr)
		 */
		void* UserParameters;
	};

	/**
	 * @brief Initializes the inference parameters.
	 * @param _Input The inference parameters.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitInferenceParameters)(
		_Dragonian_Lib_Svc_Add_Prefix(InferenceParameters)* _Input
		);

	/**
	 * @brief F0 extractor parameters
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorParameters)
	{
		INT32 SamplingRate;	///< Sampling rate
		INT32 HopSize;	///< Hop size
		INT32 F0Bins;	///< F0 bins
		INT32 WindowSize; ///< Window size
		double F0Max;	///< F0 max
		double F0Min;	///< F0 min
		float Threshold; ///< F0 threshold
		void* UserParameter;	///< User parameter
	};

	/**
	 * @brief Cluster configuration
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(ClusterConfig)
	{
		INT64 ClusterCenterSize;  ///< Cluster center size
		LPCWSTR Path;              ///< Path
		LPCWSTR Type;              ///< Type ("KMeans" or "Index")
	};

	/**
	 * @brief Slicer settings
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(SlicerSettings)
	{
		INT32 SamplingRate;	///< Sampling rate
		double Threshold; ///< Mute Threshold
		double MinLength; ///< Minimum length
		INT32 WindowLength; ///< Window length
		INT32 HopSize; ///< Hop size
	};


#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	/**
	 * @brief Initializes the F0 extractor settings.
	 * @param _Input The F0 extractor settings.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitF0ExtractorParameters)(
		_Dragonian_Lib_Svc_Add_Prefix(F0ExtractorParameters)* _Input
		);

	/**
	 * @brief Initializes the slicer settings.
	 * @param _Input The slicer settings.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitSlicerSettings)(
		_Dragonian_Lib_Svc_Add_Prefix(SlicerSettings)* _Input
		);

	/******************************************Fun**********************************************/

	/**
	 * @brief Get the last error message.
	 * @return The last error message. (Should be freed with _Dragonian_Lib_Svc_Add_Prefix(FreeString))
	 */
	_Dragonian_Lib_Svc_Api BSTR _Dragonian_Lib_Svc_Add_Prefix(GetLastError)();

	/**
	 * @brief Sets the global environment directory.
	 * @param _Dir The directory path.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetGlobalEnvDir)(
		LPCWSTR _Dir
		);

	/**
	 * @brief Sets the logger ID.
	 * @param _Id The logger ID.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerId)(
		LPCWSTR _Id
		);

	/**
	 * @brief Sets the logger level.
	 * @param _Level The logger level, Info(0), Warn(1), Error(2), None(3)
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerLevel)(
		INT32 _Level
		);

	/**
	 * @brief Sets the logger function.
	 * @param _Logger The logger function.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetLogFunction)(
		_Dragonian_Lib_Svc_Add_Prefix(LogFunction) _Logger
		);

	/**
	 * @brief Initializes the library.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(Init)();

	/**
	 * @brief Frees a string.
	 * @param _String The string to free.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(FreeString)(
		BSTR _String
		);

	/**
	 * @brief Frees data.
	 * @param _Ptr The data to free.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(FreeData)(
		void* _Ptr
		);

	/**
	 * @brief Create an environment.
	 * @param _Setting Enviroment settings
	 * @return The environment. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(DestoryEnv))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Dragonian_Lib_Svc_Add_Prefix(CreateEnviroment)(
		const _Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting)* _Setting
		);

	/**
	 * @brief Destory an environment.
	 * @param _Enviroment The environment.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(DestoryEnviroment)(
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Load a model.
	 * @param _HyperParameters The model parameters.
	 * @param _Enviroment The environment.
	 * @return The model. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(UnrefModel))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Model) _Dragonian_Lib_Svc_Add_Prefix(LoadModel)(
		const _Dragonian_Lib_Svc_Add_Prefix(HyperParameters)* _HyperParameters,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Unref a model. (Decrease the reference count, model will be freed if the reference count is zero, global scope has a reference, it should be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache), so the model will not be freed until the global reference is released)
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(UnrefModel)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model
		);

	/**
	 * @brief Load vocoder model.
	 * @param _Name The name of the vocoder model.
	 * @param _Path The path of the vocoder model.
	 * @param _SamplingRate The sampling rate of the vocoder model.
	 * @param _MelBins The mel bins of the vocoder model.
	 * @param _Enviroment The environment.
	 * @return The vocoder model. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(UnrefVocoder))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Vocoder) _Dragonian_Lib_Svc_Add_Prefix(LoadVocoder)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _SamplingRate,
		INT64 _MelBins,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Unref a model. (Decrease the reference count, model will be freed if the reference count is zero, global scope has a reference, it should be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache), so the model will not be freed until the global reference is released)
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(UnrefVocoder)(
		_Dragonian_Lib_Svc_Add_Prefix(Vocoder) _Model
		);

	/**
	 * @brief Load a units encoder model.
	 * @param _Name The name of the units encoder model.
	 * @param _Path The path of the units encoder model.
	 * @param _SamplingRate The sampling rate of the units encoder model.
	 * @param _UnitDims The unit dimensions of the units encoder model.
	 * @param _Enviroment The environment.
	 * @return The units encoder model. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(UnrefUnitsEncoder))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Dragonian_Lib_Svc_Add_Prefix(LoadUnitsEncoder)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _SamplingRate,
		INT64 _UnitDims,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Unref a model. (Decrease the reference count, model will be freed if the reference count is zero, global scope has a reference, it should be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache), so the model will not be freed until the global reference is released)
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(UnrefUnitsEncoder)(
		_Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Model
		);

	/**
	 * @brief Release a global reference. (Release the global reference, model will be freed if the reference count is zero, if there still has an active reference, the model will not be freed)
	 * @param _ModelPath The model path. (The same path you used to load the model, if a model has multiple paths, you should call this function with each path)
	 * @param _Enviroment The environment. (The same environment you used to load the model)
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache)(
		LPCWSTR _ModelPath,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Release all global reference. (Release all global reference, all models will be freed if the reference count is zero, if there still has an active reference, the model will not be freed)
	 * @param _Enviroment The environment. (The same environment you used to load the model)
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(ClearGlobalCache)(
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		);

	/**
	 * @brief Create a cluster.
	 * @param _Name The name of the cluster.
	 * @param _Path The path of the cluster.
	 * @param _ClusterDimension The dimension of the cluster.
	 * @param _ClusterSize The size of the cluster.
	 * @return The cluster. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(DestoryCluster))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Cluster) _Dragonian_Lib_Svc_Add_Prefix(CreateCluster)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _ClusterDimension,
		INT64 _ClusterSize
		);

	/**
	 * @brief Destory a cluster. 
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(DestoryCluster)(
		_Dragonian_Lib_Svc_Add_Prefix(Cluster) _Model
		);

	/**
	 * @brief Create a F0 extractor.
	 * @param _Name The name of the F0 extractor.
	 * @param _Path The path of the F0 extractor.
	 * @param _Enviroment The environment.
	 * @param _SamplingRate The sampling rate of the F0 extractor.
	 * @return The F0 extractor. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(UnrefF0Extractor))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Dragonian_Lib_Svc_Add_Prefix(CreateF0Extractor)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment,
		INT64 _SamplingRate
		);

	/**
	 * @brief Unref a F0 extractor. (If is not rmvpe or fcpe, will be destoryed, decrease the reference count, model will be freed if the reference count is zero, global scope has a reference, it should be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache), so the model will not be freed until the global reference is released)
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(UnrefF0Extractor)(
		_Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Model
		);

	/**
	 * @brief Create a tensor, shape is [_0, _1, _2, _3]
	 * @return 4d float tensor
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(CreateFloatTensor)(
		float* _Buffer,
		INT64 _0,
		INT64 _1,
		INT64 _2,
		INT64 _3
		);

	/**
	 * @brief Get data buffer of a tensor
	 * @param _Tensor Tensor to get data buffer
	 * @return Data buffer of the tensor
	 */
	_Dragonian_Lib_Svc_Api float* _Dragonian_Lib_Svc_Add_Prefix(GetTensorData)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
		);

	/**
	 * @brief Get shape of a tensor
	 * @param _Tensor Tensor to get shape
	 * @return Shape of the tensor, the shape is a 4d array, the value of each element is the size of each dimension
	 */
	_Dragonian_Lib_Svc_Api const INT64* _Dragonian_Lib_Svc_Add_Prefix(GetTensorShape)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
		);

	/**
	 * @brief Destory a tensor
	 * @param _Tensor Tensor to destory
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(DestoryFloatTensor)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
		);

	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(EncodeUnits)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
		INT64 _SourceSamplingRate,
		_Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Model
		);

	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(ClusterSearch)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Units,
		INT64 _CodeBookId,
		_Dragonian_Lib_Svc_Add_Prefix(Cluster) _Model
		);

	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(ExtractF0)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
		const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorParameters)* _Parameters,
		_Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Model
		);

	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(Inference)(
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
		INT64 SourceSamplingRate,
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Units,
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _F0,
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _SpeakerMix,
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Spec,
		const _Dragonian_Lib_Svc_Add_Prefix(InferenceParameters)* _Parameters,
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
		_Dragonian_Lib_Svc_Add_Prefix(FloatTensor)* _OutF0
		);

#ifdef __cplusplus
}
#endif