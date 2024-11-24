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

#ifdef __GNUC__
#define LibSvcDeprecated __attribute__((deprecated))
#else
#ifdef _MSC_VER
#define LibSvcDeprecated __declspec(deprecated)
#endif
#endif
#ifdef _WIN32
#include "wtypes.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
	typedef signed char         INT8, * PINT8;
	typedef signed short        INT16, * PINT16;
	typedef signed int          INT32, * PINT32;
	typedef signed long long      INT64, * PINT64;
	typedef unsigned char       UINT8, * PUINT8;
	typedef unsigned short      UINT16, * PUINT16;
	typedef unsigned int        UINT32, * PUINT32;
	typedef unsigned long long    UINT64, * PUINT64;
	typedef wchar_t* NWPSTR, * LPWSTR, * PWSTR, * BSTR;
	typedef const wchar_t* LPCWSTR;
#endif

	typedef void(*ProgCallback)(size_t cur, size_t total);			///< Progress callback function 
	typedef struct ___LIBSVCAPIT1___* LibSvcFloatVector;			///< Vector<float>
	typedef struct ___LIBSVCAPIT2___* LibSvcDoubleDimsFloatVector;	///< Vector<Vector<float>>
	typedef struct ___LIBSVCAPIT3___* LibSvcInt16Vector;			///< Vector<int16_t>
	typedef struct ___LIBSVCAPIT4___* LibSvcUInt64Vector;			///< Vector<size_t>
	typedef struct ___LIBSVCAPIT5___* LibSvcMelType;				///< Pair<Vector<float>, int64_t>
	typedef struct ___LIBSVCAPIT6___* LibSvcSliceType;				///< MoeVoiceStudioSvcSlice
	typedef struct ___LIBSVCAPIT7___* LibSvcSlicesType;				///< Array Of Slice
	typedef struct ___LIBSVCAPIT8___* LibSvcModel;					///< SharedPtr<OrtSession>
	typedef struct ___LIBSVCAPIT9___* LibSvcVocoderModel;			///< SharedPtr<OrtSession>
	typedef struct ___LIBSVCAPIT10___* LibSvcEnv;					///< SharedPtr<DragonianLibEnv>
	typedef const ___LIBSVCAPIT1___* LibSvcCFloatVector;			///< const Vector<float>
	typedef const ___LIBSVCAPIT2___* LibSvcCDoubleDimsFloatVector;	///< const Vector<Vector<float>>
	typedef const ___LIBSVCAPIT3___* LibSvcCInt16Vector;			///< const Vector<int16_t>
	typedef const ___LIBSVCAPIT4___* LibSvcCUInt64Vector;			///< const Vector<size_t>
	typedef const ___LIBSVCAPIT5___* LibSvcCMelType;				///< const Pair<Vector<float>, int64_t>
	typedef const ___LIBSVCAPIT6___* LibSvcCSliceType;				///< const MoeVoiceStudioSvcSlice
	typedef const ___LIBSVCAPIT7___* LibSvcCSlicesType;				///< const Array Of Slice
	typedef const ___LIBSVCAPIT10___* LibSvcCEnv;					///< const SharedPtr<DragonianLibEnv>

	typedef void(*LibSvcLoggerFunction)(unsigned Level, const wchar_t* Message, const wchar_t* Id); ///< Logger function

	///< Execution Providers(0:CPU, 1:CUDA, 2:DML)
	enum LibSvcExecutionProviders
	{
		CPU = 0,
		CUDA = 1,
		DML = 2
	};

	///< Model Type(0:Vits, 1:Diffusion, 2:Reflow)
	enum LibSvcModelType
	{
		Vits,
		Diffusion,
		Reflow
	};

#ifdef _MSC_VER
#pragma pack(push, 4)
#else
#pragma pack(4)
#endif

	/**
	 * @struct LibSvcSlicerSettings
	 * @brief Slicer settings
	 */
	struct LibSvcSlicerSettings
	{
		INT32 SamplingRate;	///< Sampling rate
		double Threshold; ///< Mute Threshold
		double MinLength; ///< Minimum length
		INT32 WindowLength; ///< Window length
		INT32 HopSize; ///< Hop size
	};

	/**
	 * @struct LibSvcParams
	 * @brief Inference parameters
	 */
	struct LibSvcParams
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
		LibSvcVocoderModel VocoderModel;	///< Vocoder model
		INT32 VocoderHopSize;				///< Vocoder hop size
		INT32 VocoderMelBins;		///< Vocoder mel bins
		INT32 VocoderSamplingRate;		///< Vocoder sampling rate
		long F0Bins;	///< F0 bins		
		double F0Max;  ///< F0 max
		double F0Min;   ///< F0 min
		void* F0ExtractorUserParameter;   ///< F0 extractor user parameter
		INT32 __DEBUG__MODE__;		///< Debug mode					[0:False 1:True]	
	};

	/**
	 * @struct LibSvcF0ExtractorSetting
	 * @brief F0 extractor settings
	 */
	struct LibSvcF0ExtractorSetting
	{
		INT32 SamplingRate;	///< Sampling rate
		INT32 HopSize;	///< Hop size
		INT32 F0Bins;	///< F0 bins
		double F0Max;	///< F0 max
		double F0Min;	///< F0 min
		void* UserParameter;	///< User parameter
		LibSvcEnv Env;	///< Environment
		const wchar_t* ModelPath;	///< Model path
	};

	/**
	 * @struct DiffusionSvcPaths
	 * @brief Diffusion Svc paths
	 */
	struct DiffusionSvcPaths
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
	 * @struct ReflowSvcPaths
	 * @brief Reflow Svc paths
	 */
	struct ReflowSvcPaths
	{
		LPWSTR Encoder;     ///< Encoder path
		LPWSTR VelocityFn;  ///< Velocity function path
		LPWSTR After;       ///< After path
	};

	/**
	 * @struct VitsSvcPaths
	 * @brief Vits Svc paths
	 */
	struct VitsSvcPaths
	{
		LPWSTR VitsSvc;  ///< Vits Svc path
	};

	/**
	 * @struct LibSvcClusterConfig
	 * @brief Cluster configuration
	 */
	struct LibSvcClusterConfig
	{
		INT64 ClusterCenterSize;  ///< Cluster center size
		LPWSTR Path;              ///< Path
		LPWSTR Type;              ///< Type ("KMeans" or "Index")
	};

	/**
	 * @struct LibSvcHparams
	 * @brief Model parameters
	 */
	struct LibSvcHparams
	{
		LPWSTR TensorExtractor;	 ///< Tensor extractor path
		LPWSTR HubertPath;       ///< Hubert path
		DiffusionSvcPaths DiffusionSvc;  ///< Diffusion Svc paths
		VitsSvcPaths VitsSvc;            ///< Vits Svc paths
		ReflowSvcPaths ReflowSvc;        ///< Reflow Svc paths
		LibSvcClusterConfig Cluster;     ///< Cluster configuration

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
		float Scale;    ///< Scale
	};


#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	/**
	 * @brief Initialize the LibSvcHparams structure
	 * @param _Input LibSvcHparams structure
	 */
	LibSvcApi void InitLibSvcHparams(
		LibSvcHparams* _Input
	);

	/**
	 * @brief Initialize the LibSvcParams structure
	 * @param _Input LibSvcParams structure
	 */
	LibSvcApi void InitLibSvcParams(
		LibSvcParams* _Input
	);

	/**
	 * @brief Initialize the LibSvcF0ExtractorSetting structure
	 * @param _Input LibSvcF0ExtractorSetting structure
	 */
	LibSvcApi void InitLibSvcF0ExtractorSetting(
		LibSvcF0ExtractorSetting* _Input
	);

	/**
	 * @brief Initialize the LibSvcSlicerSettings structure
	 * @param _Input LibSvcSlicerSettings structure
	 */
	LibSvcApi void InitLibSvcSlicerSettings(
		LibSvcSlicerSettings* _Input
	);

	/**
	 * @brief Get the buffer of the float vector
	 * @param _Obj Float vector
	 * @return Buffer of the float vector
	 */
	LibSvcApi float* LibSvcGetFloatVectorData(
		LibSvcFloatVector _Obj
	);

	/**
	 * @brief Get the size of the float vector
	 * @param _Obj Float vector
	 * @return Size of the float vector
	 */
	LibSvcApi size_t LibSvcGetFloatVectorSize(
		LibSvcFloatVector _Obj
	);

	/**
	 * @brief Allocate a float vector
	 * @return A new float vector
	 */
	LibSvcApi LibSvcFloatVector LibSvcAllocateFloatVector();

	/**
	 * @brief Release the float vector
	 * @param _Obj Float vector
	 */
	LibSvcApi void LibSvcReleaseFloatVector(
		LibSvcFloatVector _Obj
	);

	/**
	 * @brief Insert _ObjB into _ObjA
	 * @param _ObjA Inserted float vector
	 * @param _ObjB Inserting float vector
	 */
	LibSvcApi void LibSvcInsertFloatVector(
		LibSvcFloatVector _ObjA,
		LibSvcFloatVector _ObjB
	);

	/**
	 * @brief Get the float vector from the double-dimension float vector
	 * @param _Obj Double-dimension float vector
	 * @param _Index Index of the float vector
	 * @return Float vector
	 */
	LibSvcApi LibSvcFloatVector LibSvcGetDFloatVectorData(
		LibSvcDoubleDimsFloatVector _Obj,
		size_t _Index
	);

	/**
	 * @brief Get the size of the double-dimension float vector
	 * @param _Obj Double-dimension float vector
	 * @return Size of the double-dimension float vector
	 */
	LibSvcApi size_t LibSvcGetDFloatVectorSize(
		LibSvcDoubleDimsFloatVector _Obj
	);

	/**
	 * @brief Allocate a new int16 vector
	 * @return A new int16 vector
	 */
	LibSvcApi LibSvcInt16Vector LibSvcAllocateInt16Vector();

	/**
	 * @brief Release the int16 vector
	 * @param _Obj The int16 vector to release
	 */
	LibSvcApi void LibSvcReleaseInt16Vector(
		LibSvcInt16Vector _Obj
	);

	/**
	 * @brief Set the length of the int16 vector
	 * @param _Obj The int16 vector
	 * @param _Size The new size of the vector
	 */
	LibSvcApi void LibSvcSetInt16VectorLength(
		LibSvcInt16Vector _Obj,
		size_t _Size
	);

	/**
	 * @brief Insert one int16 vector into another
	 * @param _ObjA The target int16 vector
	 * @param _ObjB The int16 vector to insert
	 */
	LibSvcApi void LibSvcInsertInt16Vector(
		LibSvcInt16Vector _ObjA,
		LibSvcInt16Vector _ObjB
	);

	/**
	 * @brief Get the data of the int16 vector
	 * @param _Obj The int16 vector
	 * @return Pointer to the data of the int16 vector
	 */
	LibSvcApi short* LibSvcGetInt16VectorData(
		LibSvcInt16Vector _Obj
	);

	/**
	 * @brief Get the size of the int16 vector
	 * @param _Obj The int16 vector
	 * @return Size of the int16 vector
	 */
	LibSvcApi size_t LibSvcGetInt16VectorSize(
		LibSvcInt16Vector _Obj
	);

	/**
	 * @brief Allocate a new uint64 vector
	 * @return A new uint64 vector
	 */
	LibSvcApi LibSvcUInt64Vector LibSvcAllocateUInt64Vector();

	/**
	 * @brief Release the uint64 vector
	 * @param _Obj The uint64 vector to release
	 */
	LibSvcApi void LibSvcReleaseUInt64Vector(
		LibSvcUInt64Vector _Obj
	);

	/**
	 * @brief Set the length of the uint64 vector
	 * @param _Obj The uint64 vector
	 * @param _Size The new size of the vector
	 */
	LibSvcApi void LibSvcSetUInt64VectorLength(
		LibSvcUInt64Vector _Obj,
		size_t _Size
	);

	/**
	 * @brief Get the data of the uint64 vector
	 * @param _Obj The uint64 vector
	 * @return Pointer to the data of the uint64 vector
	 */
	LibSvcApi size_t* LibSvcGetUInt64VectorData(
		LibSvcUInt64Vector _Obj
	);

	/**
	 * @brief Get the size of the uint64 vector
	 * @param _Obj The uint64 vector
	 * @return Size of the uint64 vector
	 */
	LibSvcApi size_t LibSvcGetUInt64VectorSize(
		LibSvcUInt64Vector _Obj
	);

	/**
	 * @brief Allocate a new Mel type
	 * @return A new Mel type
	 */
	LibSvcApi LibSvcMelType LibSvcAllocateMel();

	/**
	 * @brief Release the Mel type
	 * @param _Obj The Mel type to release
	 */
	LibSvcApi void LibSvcReleaseMel(
		LibSvcMelType _Obj
	);

	/**
	 * @brief Get the data of the Mel type
	 * @param _Obj The Mel type
	 * @return Float vector containing the Mel data
	 */
	LibSvcApi LibSvcFloatVector LibSvcGetMelData(
		LibSvcMelType _Obj
	);

	/**
	 * @brief Get the size of the Mel type
	 * @param _Obj The Mel type
	 * @return Size of the Mel type
	 */
	LibSvcApi INT64 LibSvcGetMelSize(
		LibSvcMelType _Obj
	);

	/**
	 * @brief Get the audio data from the slice type
	 * @param _Obj The slice type
	 * @return Float vector containing the audio data
	 */
	LibSvcApi LibSvcFloatVector LibSvcGetAudio(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Get the F0 data from the slice type
	 * @param _Obj The slice type
	 * @return Float vector containing the F0 data
	 */
	LibSvcApi LibSvcFloatVector LibSvcGetF0(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Get the volume data from the slice type
	 * @param _Obj The slice type
	 * @return Float vector containing the volume data
	 */
	LibSvcApi LibSvcFloatVector LibSvcGetVolume(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Get the speaker data from the slice type
	 * @param _Obj The slice type
	 * @return Double-dimension float vector containing the speaker data
	 */
	LibSvcApi LibSvcDoubleDimsFloatVector LibSvcGetSpeaker(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Get the source length from the slice type
	 * @param _Obj The slice type
	 * @return Source length
	 */
	LibSvcApi UINT64 LibSvcGetSrcLength(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Check if the slice type is not mute
	 * @param _Obj The slice type
	 * @return 1 if not mute, 0 otherwise
	 */
	LibSvcApi INT32 LibSvcGetIsNotMute(
		LibSvcSliceType _Obj
	);

	/**
	 * @brief Set the size of the speaker mix data
	 * @param _Obj The slice type
	 * @param _NSpeaker The number of speakers
	 */
	LibSvcApi void LibSvcSetSpeakerMixDataSize(
		LibSvcSliceType _Obj,
		size_t _NSpeaker
	);

	/**
	 * @brief Allocate new slice data
	 * @return New slice data
	 */
	LibSvcApi LibSvcSlicesType LibSvcAllocateSliceData();

	/**
	 * @brief Release the slice data
	 * @param _Obj The slice data to release
	 */
	LibSvcApi void LibSvcReleaseSliceData(
		LibSvcSlicesType _Obj
	);

	/**
	 * @brief Get the audio path from the slice data
	 * @param _Obj The slice data
	 * @return Audio path as a BSTR
	 */
	LibSvcApi BSTR LibSvcGetAudioPath(
		LibSvcSlicesType _Obj
	);

	/**
	 * @brief Get a slice from the slice data
	 * @param _Obj The slice data
	 * @param _Index The index of the slice
	 * @return The slice at the specified index
	 */
	LibSvcApi LibSvcSliceType LibSvcGetSlice(
		LibSvcSlicesType _Obj,
		size_t _Index
	);

	/**
	 * @brief Get the count of slices in the slice data
	 * @param _Obj The slice data
	 * @return Count of slices
	 */
	LibSvcApi size_t LibSvcGetSliceCount(
		LibSvcSlicesType _Obj
	);

	/******************************************Fun**********************************************/

	/**
	 * @brief Sets the global environment directory.
	 * @param _Dir The directory path.
	 */
	LibSvcApi void LibSvcSetGlobalEnvDir(
		LPCWSTR _Dir
	);

	/**
	 * @brief Sets the logger ID.
	 * @param _Id The logger ID.
	 */
	LibSvcApi void LibSvcSetLoggerId(
		LPCWSTR _Id
	);

	/**
	 * @brief Sets the logger level.
	 * @param _Level The logger level.
	 */
	LibSvcApi void LibSvcSetLoggerLevel(
		INT32 _Level
	);

	/**
	 * @brief Sets the logger function.
	 * @param _Logger The logger function.
	 */
	LibSvcApi void LibSvcSetLoggerFunction(
		LibSvcLoggerFunction _Logger
	);

	/**
	 * @brief Initializes the library.
	 */
	LibSvcApi void LibSvcInit();

	/**
	 * @brief Frees a string.
	 * @param _String The string to free.
	 */
	LibSvcApi void LibSvcFreeString(
		BSTR _String
	);

	/**
	 * @brief Creates an environment.
	 * @param ThreadCount The number of threads.
	 * @param DeviceID The device ID.
	 * @param Provider The provider.
	 * @return The created environment.
	 */
	LibSvcApi LibSvcEnv LibSvcCreateEnv(
		UINT32 ThreadCount,
		UINT32 DeviceID,
		UINT32 Provider
	);

	/**
	 * @brief Destroys an environment.
	 * @param Env The environment to destroy.
	 */
	LibSvcApi void LibSvcDestoryEnv(
		LibSvcEnv Env
	);

	/**
	 * @brief Slices audio data (int16 version).
	 * @param _Audio The audio data.
	 * @param _Setting The slicer settings.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcSliceAudioI16(
		LibSvcCInt16Vector _Audio,
		const LibSvcSlicerSettings* _Setting,
		LibSvcUInt64Vector _Output
	);

	/**
	 * @brief Slices audio data (float version).
	 * @param _Audio The audio data.
	 * @param _Setting The slicer settings.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcSliceAudio(
		LibSvcCFloatVector _Audio,
		const LibSvcSlicerSettings* _Setting,
		LibSvcUInt64Vector _Output
	);

	/**
	 * @brief Preprocesses audio data (int16 version).
	 * @param _Audio The audio data.
	 * @param _SlicePos The slice positions.
	 * @param _Settings The settings of the F0 extractor.
	 * @param _Threshold The threshold.
	 * @param _F0Method The F0 method.
	 * @param _Output The output slices.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcPreprocessI16(
		LibSvcCInt16Vector _Audio,
		LibSvcCUInt64Vector _SlicePos,
		const LibSvcF0ExtractorSetting* _Settings,
		double _Threshold,
		const wchar_t* _F0Method,
		LibSvcSlicesType _Output
	);

	/**
	 * @brief Preprocesses audio data (float version).
	 * @param _Audio The audio data.
	 * @param _SlicePos The slice positions.
	 * @param _Settings The settings of the F0 extractor.
	 * @param _Threshold The threshold.
	 * @param _F0Method The F0 method.
	 * @param _Output The output slices.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcPreprocess(
		LibSvcCFloatVector _Audio,
		LibSvcCUInt64Vector _SlicePos,
		const LibSvcF0ExtractorSetting* _Settings,
		double _Threshold,
		const wchar_t* _F0Method,
		LibSvcSlicesType _Output
	);

	/**
	 * @brief Performs STFT on audio data (int16 version).
	 * @param _Audio The audio data.
	 * @param _SamplingRate The sampling rate.
	 * @param _Hopsize The hop size.
	 * @param _MelBins The number of mel bins.
	 * @param _Output The output mel type.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcStftI16(
		LibSvcCInt16Vector _Audio,
		INT32 _SamplingRate,
		INT32 _Hopsize,
		INT32 _MelBins,
		LibSvcMelType _Output
	);

	/**
	 * @brief Performs STFT on audio data (float version).
	 * @param _Audio The audio data.
	 * @param _SamplingRate The sampling rate.
	 * @param _Hopsize The hop size.
	 * @param _MelBins The number of mel bins.
	 * @param _Output The output mel type.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcStft(
		LibSvcCFloatVector _Audio,
		INT32 _SamplingRate,
		INT32 _Hopsize,
		INT32 _MelBins,
		LibSvcMelType _Output
	);

	/**
	 * @brief Infers a slice.
	 * @param _Model The model.
	 * @param _Slice The slice.
	 * @param _InferParams The inference parameters.
	 * @param _Process The process.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcInferSlice(
		LibSvcModel _Model,
		LibSvcCSliceType _Slice,
		const LibSvcParams* _InferParams,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Infers audio data.
	 * @param _Model The model.
	 * @param _Audio The audio slices.
	 * @param _InferParams The inference parameters.
	 * @param _SrcLength The source length.
	 * @param _Process The process.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcInferAudio(
		LibSvcModel _Model,
		LibSvcSlicesType _Audio,
		const LibSvcParams* _InferParams,
		UINT64 _SrcLength,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Infers PCM data.
	 * @param _Model The model.
	 * @param _PCMData The PCM data.
	 * @param _InferParams The inference parameters.
	 * @param SamplingRate The sampling rate.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcInferPCMData(
		LibSvcModel _Model,
		LibSvcCFloatVector _PCMData,
		const LibSvcParams* _InferParams,
		INT32 SamplingRate,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Infers shallow diffusion.
	 * @param _Model The model.
	 * @param _16KAudioHubert The 16K audio Hubert.
	 * @param _Mel The mel type.
	 * @param _SrcF0 The source F0.
	 * @param _SrcVolume The source volume.
	 * @param _SrcSpeakerMap The source speaker map.
	 * @param _SrcSize The source size.
	 * @param _InferParams The inference parameters.
	 * @param _Process The process.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcShallowDiffusionInference(
		LibSvcModel _Model,
		LibSvcFloatVector _16KAudioHubert,
		LibSvcMelType _Mel,
		LibSvcCFloatVector _SrcF0,
		LibSvcCFloatVector _SrcVolume,
		LibSvcCDoubleDimsFloatVector _SrcSpeakerMap,
		INT64 _SrcSize,
		const LibSvcParams* _InferParams,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Vocoder Enhance.
	 * @param _Model Vocoder model.
	 * @param _Env The environment.
	 * @param _Mel The mel type.
	 * @param _F0 The F0.
	 * @param _VocoderMelBins The vocoder mel bins.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcVocoderEnhance(
		LibSvcVocoderModel _Model,
		LibSvcEnv _Env,
		LibSvcMelType _Mel,
		LibSvcCFloatVector _F0,
		INT32 _VocoderMelBins,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Loads a model.
	 * @param _T The model type.
	 * @param _Config The model configuration.
	 * @param _ProgressCallback The progress callback.
	 * @param _ExecutionProvider The execution provider.
	 * @param _DeviceID The device ID.
	 * @param _ThreadCount The number of threads.
	 * @return The loaded model.
	 */
	LibSvcApi LibSvcModel LibSvcLoadModel(
		UINT32 _T,
		const LibSvcHparams* _Config,
		ProgCallback _ProgressCallback,
		UINT32 _ExecutionProvider,
		UINT32 _DeviceID,
		UINT32 _ThreadCount
	);

	/**
	 * @brief Unloads a model.
	 * @param _Model The model to unload.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcUnloadModel(
		LibSvcModel _Model
	);

	/**
	 * @brief Loads a vocoder model.
	 * @param VocoderPath The vocoder path.
	 * @param _Env The environment.
	 * @return The loaded vocoder model.
	 */
	LibSvcApi LibSvcVocoderModel LibSvcLoadVocoder(
		LPCWSTR VocoderPath,
		LibSvcEnv _Env
	);

	/**
	 * @brief Unloads a ort model.
	 * @param ModelPath The model path.
	 * @param _Env The environment.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcUnloadCachedModel(
		LPCWSTR ModelPath,
		LibSvcEnv _Env
	);

	/**
	 * @brief Unloads all cached models.
	 */
	LibSvcApi void LibSvcClearCachedModel();

	/**
	 * @brief Reads audio data from a file.
	 * @param _AudioPath The audio file path.
	 * @param _SamplingRate The sampling rate.
	 * @param _Output The output vector.
	 * @return Status code.
	 */
	LibSvcApi INT32 LibSvcReadAudio(
		LPCWSTR _AudioPath,
		INT32 _SamplingRate,
		LibSvcFloatVector _Output
	);

	/**
	 * @brief Writes audio data to a file.
	 * @param _PCMData The PCM data.
	 * @param _OutputPath The output file path.
	 * @param _SamplingRate The sampling rate.
	 */
	LibSvcApi void LibSvcWriteAudioFile(
		LibSvcFloatVector _PCMData,
		LPCWSTR _OutputPath,
		INT32 _SamplingRate
	);

#ifdef __cplusplus
}
#endif