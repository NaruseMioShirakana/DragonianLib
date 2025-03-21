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

	
	typedef struct ____Dragonian_Lib_Svc_ApiT1___* _Dragonian_Lib_Svc_Add_Prefix(Model); ///< SingingVoiceConversion*
	typedef struct ____Dragonian_Lib_Svc_ApiT2___* _Dragonian_Lib_Svc_Add_Prefix(VocoderModel); ///< SharedPtr<OrtSession>
	typedef struct ____Dragonian_Lib_Svc_ApiT3___* _Dragonian_Lib_Svc_Add_Prefix(Env); ///< SharedPtr<DragonianLibEnv>
	typedef struct ____Dragonian_Lib_Svc_ApiT4___* _Dragonian_Lib_Svc_Add_Prefix(InferenceData); ///< Inference Data
	typedef struct ____Dragonian_Lib_Svc_ApiT5___* _Dragonian_Lib_Svc_Add_Prefix(Slice); ///< Slice
	typedef struct ____Dragonian_Lib_Svc_ApiT6___* _Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData); ///< Speaker Mix Data

	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(ProgressCallback))(size_t cur, size_t total); ///< Progress callback function 
	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(LoggerFunction))(unsigned Level, const wchar_t* Message, const wchar_t* Id); ///< Logger function
	typedef void(*_Dragonian_Lib_Svc_Add_Prefix(Deleter))(void*); ///< Deleter function

	///< Execution Providers(0:CPU, 1:CUDA, 2:DML)
	enum _Dragonian_Lib_Svc_Add_Prefix(ExecutionProvider)
	{
		_Dragonian_Lib_Svc_Add_Prefix(CPUEP) = 0, _Dragonian_Lib_Svc_Add_Prefix(CUDAEP) = 1, _Dragonian_Lib_Svc_Add_Prefix(DMLEP) = 2
	};

#ifdef _MSC_VER
#pragma pack(push, 4)
#else
#pragma pack(4)
#endif

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

	/**
	 * @brief Inference parameters
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(Params)
	{
		float NoiseScale;	///< Noise scale factor					[0 ~ 10]
		INT64 Seed;			///< Random seed						[INT64]
		INT64 SpeakerId;	///< Speaker ID							[INT64]
		INT64 SpkCount;		///< Speaker count						[INT64]
		float IndexRate;	///< Index rate							[0 ~ 1]
		float ClusterRate;	///< Cluster rate						[0 ~ 1]
		float DDSPNoiseScale;		///< DDSP noise scale			[0 ~ 10]
		float Keys;					///< Keys						[0 ~ 1]
		size_t MeanWindowLength;	///< Mean window length			[1 ~ 1000]
		size_t Pndm;				///< Diffusion Skip Num			[1 ~ Step]
		size_t Step;	///< Diffusion Step							[1 ~ MaxStep]
		float TBegin;	///< Reflow begin point						[0 ~ 1]
		float TEnd;		///< Reflow end point						[TBegin ~ 1]
		LPWSTR Sampler;			///< Diffusion Sampler				["Pndm" "DDim"]
		LPWSTR ReflowSampler;	///< Reflow Sampler					["Eular" "Rk4" "Heun" "Pecece"]
		LPWSTR F0Method;					///< F0 Method			["Dio" "Harvest" "RMVPE" "FCPE"]
		_Dragonian_Lib_Svc_Add_Prefix(VocoderModel) VocoderModel;	///< Vocoder model
		INT32 VocoderHopSize;				///< Vocoder hop size
		INT32 VocoderMelBins;		///< Vocoder mel bins
		INT32 VocoderSamplingRate;		///< Vocoder sampling rate
		long F0Bins;	///< F0 bins		
		double F0Max;  ///< F0 max
		double F0Min;   ///< F0 min
		void* F0ExtractorUserParameter;   ///< F0 extractor user parameter
		float MelFactor;	///< Mel factor
		INT32 __DEBUG__MODE__;		///< Debug mode					[0:False 1:True]	
	};

	/**
	 * @brief F0 extractor settings
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)
	{
		INT32 SamplingRate;	///< Sampling rate
		INT32 HopSize;	///< Hop size
		INT32 F0Bins;	///< F0 bins
		INT32 WindowSize; ///< Window size
		double F0Max;	///< F0 max
		double F0Min;	///< F0 min
		void* UserParameter;	///< User parameter
		_Dragonian_Lib_Svc_Add_Prefix(Env) Env;	///< Environment
		const wchar_t* ModelPath;	///< Model path
	};

	/**
	 * @brief Diffusion Svc paths
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(DiffusionSvcPaths)
	{
		LPWSTR Encoder;   ///< Encoder path
		LPWSTR Denoise;   ///< Denoise path
		LPWSTR Pred;      ///< Prediction path
		LPWSTR After;     ///< After path
		LPWSTR Alpha;     ///< Alpha path
		LPWSTR Naive;     ///< Naive path

		LPWSTR DiffSvc;   ///< Old Diffusion path
	};

	/**
	 * @brief Reflow Svc paths
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(ReflowSvcPaths)
	{
		LPWSTR Encoder;     ///< Encoder path
		LPWSTR VelocityFn;  ///< Velocity function path
		LPWSTR After;       ///< After path
	};

	/**
	 * @brief Vits Svc paths
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(VitsSvcPaths)
	{
		LPWSTR VitsSvc;  ///< Vits Svc path
	};

	/**
	 * @brief Cluster configuration
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(ClusterConfig)
	{
		INT64 ClusterCenterSize;  ///< Cluster center size
		LPWSTR Path;              ///< Path
		LPWSTR Type;              ///< Type ("KMeans" or "Index")
	};

	/**
	 * @brief Model parameters
	 */
	struct _Dragonian_Lib_Svc_Add_Prefix(Hparams)
	{
		LPWSTR TensorExtractor;	 ///< Tensor extractor path
		LPWSTR HubertPath;       ///< Hubert path
		_Dragonian_Lib_Svc_Add_Prefix(DiffusionSvcPaths) DiffusionSvc;  ///< Diffusion Svc paths
		_Dragonian_Lib_Svc_Add_Prefix(VitsSvcPaths) VitsSvc;            ///< Vits Svc paths
		_Dragonian_Lib_Svc_Add_Prefix(ReflowSvcPaths) ReflowSvc;        ///< Reflow Svc paths
		_Dragonian_Lib_Svc_Add_Prefix(ClusterConfig) Cluster;     ///< Cluster configuration

		INT32 SamplingRate;  ///< Sampling rate

		INT32 HopSize;           ///< Hop size
		INT64 HiddenUnitKDims;   ///< Hidden unit K dimensions
		INT64 SpeakerCount;      ///< Speaker count
		INT32 EnableCharaMix;    ///< Enable character mix
		INT32 EnableVolume;      ///< Enable volume
		INT32 VaeMode;           ///< VAE mode

		INT64 MelBins;  ///< Mel bins
		INT64 Pndms;    ///< PNDMS
		INT64 MaxStep;  ///< Maximum step
		float SpecMin;  ///< Spectrum minimum
		float SpecMax;  ///< Spectrum maximum
		float F0Min;	///< F0 min
		float F0Max;	///< F0 max
		float Scale;    ///< Scale
	};


#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	/**
	 * @brief Init the model parameters.
	 * @param _Input The model parameters.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitHparams)(
		_Dragonian_Lib_Svc_Add_Prefix(Hparams)* _Input
	);

	/**
	 * @brief Initializes the inference parameters.
	 * @param _Input The inference parameters.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitInferenceParams)(
		_Dragonian_Lib_Svc_Add_Prefix(Params)* _Input
	);

	/**
	 * @brief Initializes the F0 extractor settings.
	 * @param _Input The F0 extractor settings.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(InitF0ExtractorSetting)(
		_Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Input
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
	 * @return The last error message. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeString))
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
	 * @param _Level The logger level.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerLevel)(
		INT32 _Level
	);

	/**
	 * @brief Sets the logger function.
	 * @param _Logger The logger function.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerFunction)(
		_Dragonian_Lib_Svc_Add_Prefix(LoggerFunction) _Logger
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
	 * @param _ThreadCount The thread count.
	 * @param _DeviceID The device ID.
	 * @param _Provider The execution provider.
	 * @return The environment. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(DestoryEnv))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Env) _Dragonian_Lib_Svc_Add_Prefix(CreateEnv)(
		UINT32 _ThreadCount,
		UINT32 _DeviceID,
		_Dragonian_Lib_Svc_Add_Prefix(ExecutionProvider) _Provider
	);

	/**
	 * @brief Destory an environment.
	 * @param _Env The environment.
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(DestoryEnv)(
		_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
	);

	/**
	 * @brief Load a model. (Create a global reference and increase the reference count)
	 * @param _Config The model parameters.
	 * @param _Env The environment. (Created by _Dragonian_Lib_Svc_Add_Prefix(CreateEnv))
	 * @param _ProgressCallback The progress callback.
	 * @return The model. (Should be unref by _Dragonian_Lib_Svc_Add_Prefix(UnrefModel) and global reference could be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Model) _Dragonian_Lib_Svc_Add_Prefix(LoadModel)(
		const _Dragonian_Lib_Svc_Add_Prefix(Hparams)* _Config,
		_Dragonian_Lib_Svc_Add_Prefix(Env) _Env,
		_Dragonian_Lib_Svc_Add_Prefix(ProgressCallback) _ProgressCallback
		);

	/**
	 * @brief Load a vocoder model. (Create a global reference and increase the reference count)
	 * @param VocoderPath The vocoder path.
	 * @param _Env The environment. (Created by _Dragonian_Lib_Svc_Add_Prefix(CreateEnv))
	 * @return The vocoder model. (Should be unref by _Dragonian_Lib_Svc_Add_Prefix(UnrefModel) and global reference could be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache))
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(VocoderModel) _Dragonian_Lib_Svc_Add_Prefix(LoadVocoder)(
		LPCWSTR VocoderPath,
		_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
		);

	/**
	 * @brief Unref a model. (Decrease the reference count, model will be freed if the reference count is zero, global scope has a reference, it should be released by _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache), so the model will not be freed until the global reference is released)
	 * @param _Model The model.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(UnrefModel)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model
		);

	/**
	 * @brief Release a global reference. (Release the global reference, model will be freed if the reference count is zero, if there still has an active reference, the model will not be freed)
	 * @param ModelPath The model path. (The same path you used to load the model, if a model has multiple paths, you should call this function with each path)
	 * @param _Env The environment. (The same environment you used to load the model)
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache)(
		LPCWSTR ModelPath,
		_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
		);

	/**
	 * @brief Release all global reference. (Release all global reference, all models will be freed if the reference count is zero, if there still has an active reference, the model will not be freed)
	 */
	_Dragonian_Lib_Svc_Api void _Dragonian_Lib_Svc_Add_Prefix(ClearGlobalCache)();

	/**
	 * @brief Slice the audio.
	 * @param[Input] _Setting The slicer settings.
	 * @param[Input] _Audio The audio.
	 * @param[Input] _AudioSize The audio size.
	 * @param[Output] _SlicePos The slice positions. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _SlicePosSize The slice position size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(SliceAudio)(
		const _Dragonian_Lib_Svc_Add_Prefix(SlicerSettings)* _Setting,
		const float* _Audio,
		size_t _AudioSize,
		size_t** _SlicePos,
		size_t* _SlicePosSize
		);

	/**
	 * @brief Get the InferenceData. an array of struct{ float[audio_size] audio, float[frames] f0, float[frames] volume, float[n_speaker][frames] speakermix };
	 * @param[Input] _Settings The F0 extractor settings.
	 * @param[Input] _Audio The audio.
	 * @param[Input] _AudioSize The audio size.
	 * @param[Input] _SlicePos The slice positions.
	 * @param[Input] _SlicePosSize The slice position size.
	 * @param[Input] _DbThreshold If the volume is lower than this value, the slice will be muted.
	 * @param[Input] _F0Method The F0 method.
	 * @param[Output] _OutputSlices The output slices. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(ReleaseInferenceData))
	 * @param[Output] _OutputSlicesSize The output slices size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(PreprocessInferenceData)(
		const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Settings,
		const float* _Audio,
		size_t _AudioSize,
		const size_t* _SlicePos,
		size_t _SlicePosSize,
		double _DbThreshold,
		const wchar_t* _F0Method,
		_Dragonian_Lib_Svc_Add_Prefix(InferenceData)* _OutputSlices,
		size_t* _OutputSlicesSize
		);

	/**
	 * @brief Release the inference data.
	 * @param[Input] _Data The inference data.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(ReleaseInferenceData)(
		_Dragonian_Lib_Svc_Add_Prefix(InferenceData) _Data
		);

	/**
	 * @brief Get the slice.
	 * @param[Input] _Data The inference data.
	 * @param[Input] _Index The index.
	 * @param[Output] _NumFrames The number of frames.
	 * @return The slice. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(Slice) _Dragonian_Lib_Svc_Add_Prefix(GetSlice)(
		_Dragonian_Lib_Svc_Add_Prefix(InferenceData) _Data,
		size_t _Index,
		size_t* _NumFrames
		);

	/**
	 * @brief Get the audio.
	 * @param[Input] _Slice The slice.
	 * @param[Output] _AudioSize The audio size.
	 * @return The audio. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api float* _Dragonian_Lib_Svc_Add_Prefix(GetAudio)(
		_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice,
		size_t* _AudioSize
		);

	/**
	 * @brief Get the F0 array.
	 * @param[Input] _Slice The slice.
	 * @return The F0 array. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api float* _Dragonian_Lib_Svc_Add_Prefix(GetF0)(
		_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
		);

	/**
	 * @brief Get the volume array.
	 * @param[Input] _Slice The slice.
	 * @return The volume array. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api float* _Dragonian_Lib_Svc_Add_Prefix(GetVolume)(
		_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
		);

	/**
	 * @brief Get the speaker mix array.
	 * @param[Input] _Slice The slice.
	 * @return The speaker mix. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Dragonian_Lib_Svc_Add_Prefix(GetSpeaker)(
		_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
		);

	/**
	 * @brief Reshape the speaker mix data.
	 * @param[Input] _Speaker The speaker mix data.
	 * @param[Input] _SpeakerCount The number of speakers.
	 * @param[Input] _NumFrame The number of frames.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(ReshapeSpeakerMixData)(
		_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Speaker,
		size_t _SpeakerCount,
		size_t _NumFrame
		);

	/**
	 * @brief Get the speaker mix data.
	 * @param[Input] _Speaker The speaker mix data.
	 * @param[Input] _Index The index.
	 * @return The speaker mix. (You don't need to free it)
	 */
	_Dragonian_Lib_Svc_Api float* _Dragonian_Lib_Svc_Add_Prefix(GetSpeakerData)(
		_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Speaker,
		size_t _Index
		);

	/**
	 * @brief Extract the mel-spectrogram.
	 * @param[Input] _Audio The audio.
	 * @param[Input] _AudioSize The audio size.
	 * @param[Input] _SamplingRate The sampling rate.
	 * @param[Input] _Hopsize The hop size.
	 * @param[Input] _MelBins The mel bins.
	 * @param[Output] _OutputMel The output mel-spectrogram. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputMelSize The output mel-spectrogram size. (_OutputMelSize / _MelBins = number of frames)
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(Stft)(
		const float* _Audio,
		size_t _AudioSize,
		INT32 _SamplingRate,
		INT32 _Hopsize,
		INT32 _MelBins,
		float** _OutputMel,
		size_t* _OutputMelSize
		);

	/**
	 * @brief Infer the slice.
	 * @param[Input] _Model The model.
	 * @param[Input] _InferParams The inference parameters.
	 * @param[Input] _Slice The slice.
	 * @param[Input/Output] _Process The process.
	 * @param[Output] _Output The output. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputSize The output size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(InferSlice)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
		const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
		_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice,
		size_t* _Process,
		float** _Output,
		size_t* _OutputSize
		);

	/**
	 * @brief Infer the audio. (The audio should be sliced to the same time(_SliceTime) as the slice, and crossfade after the inference)
	 * @param[Input] _Model The model.
	 * @param[Input] _InferParams The inference parameters.
	 * @param[Input] _Audio The audio.
	 * @param[Input] _AudioSize The audio size.
	 * @param[Input] _AudioSamplingRate The audio sampling rate.
	 * @param[Input] _F0Method The F0 method.
	 * @param[Input] _Settings The F0 extractor settings.
	 * @param[Input] _SliceTime The slice time.
	 * @param[Input] _CrossFadeTime The crossfade time.
	 * @param[Input] _DbThreshold The db threshold.
	 * @param[Output] _Output The output. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputSize The output size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(InferAudio)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
		const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
		const float* _Audio,
		size_t _AudioSize,
		long _AudioSamplingRate,
		const wchar_t* _F0Method,
		const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Settings,
		float _SliceTime,
		float _CrossFadeTime,
		double _DbThreshold,
		float** _Output,
		size_t* _OutputSize
		);

	/**
	 * @brief Infer the PCM data.
	 * @param[Input] _Model The model.
	 * @param[Input] _InferParams The inference parameters.
	 * @param[Input] _Audio The audio.
	 * @param[Input] _AudioSize The audio size.
	 * @param[Input] _InputSamplingRate The input sampling rate.
	 * @param[Output] _Output The output. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputSize The output size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(InferPCMData)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
		const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
		const float* _Audio,
		size_t _AudioSize,
		INT32 _InputSamplingRate,
		float** _Output,
		size_t* _OutputSize
		);

	/**
	 * @brief Shallow diffusion inference.
	 * @param[Input] _Model The model.
	 * @param[Input] _InferParams The inference parameters.
	 * @param[Input] _16KAudioHubert The 16K audio hubert. (Should be resampled to 16000Hz)
	 * @param[Input] _16KAudioSize The 16K audio size.
	 * @param[Input] _Mel The mel-spectrogram.
	 * @param[Input] _SrcMelSize The source mel size.
	 * @param[Input] _MelSize The mel size.
	 * @param[Input] _SrcF0 The source F0.
	 * @param[Input] _SrcF0Size The source F0 size.
	 * @param[Input] _SrcVolume The source volume.
	 * @param[Input] _SrcVolumeSize The source volume size.
	 * @param[Input] _SrcSpeakerMap The source speaker map.
	 * @param[Input/Output] _Process The process.
	 * @param[Output] _Output The output. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputSize The output size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(ShallowDiffusionInference)(
		_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
		const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
		const float* _16KAudioHubert,
		size_t _16KAudioSize,
		const float* _Mel,
		size_t _SrcMelSize,
		size_t _MelSize,
		const float* _SrcF0,
		size_t _SrcF0Size,
		const float* _SrcVolume,
		size_t _SrcVolumeSize,
		_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _SrcSpeakerMap,
		size_t* _Process,
		float** _Output,
		size_t* _OutputSize
	);

	/**
	 * @brief Infer the mel-spectrogram.
	 * @param[Input] _Model The model.
	 * @param[Input] _Env The environment.
	 * @param[Input] _Mel The mel-spectrogram.
	 * @param[Input] _SrcMelSize The source mel size.
	 * @param[Input] _SrcF0 The source F0.
	 * @param[Input] _SrcF0Size The source F0 size.
	 * @param[Input] _VocoderMelBins The vocoder mel bins.
	 * @param[Output] _Output The output. (Should be freed by _Dragonian_Lib_Svc_Add_Prefix(FreeData))
	 * @param[Output] _OutputSize The output size.
	 * @return 0 if succeeded, otherwise 1.
	 */
	_Dragonian_Lib_Svc_Api INT32 _Dragonian_Lib_Svc_Add_Prefix(VocoderEnhance)(
		_Dragonian_Lib_Svc_Add_Prefix(VocoderModel) _Model,
		_Dragonian_Lib_Svc_Add_Prefix(Env) _Env,
		const float* _Mel,
		size_t _SrcMelSize,
		const float* _SrcF0,
		size_t _SrcF0Size,
		INT32 _VocoderMelBins,
		float** _Output,
		size_t* _OutputSize
	);

#ifdef __cplusplus
}
#endif