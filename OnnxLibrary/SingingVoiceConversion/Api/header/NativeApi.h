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
#include "DynLibExport.h"

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
#endif

	typedef void(*ProgCallback)(size_t, size_t);
	typedef struct ___LIBSVCAPIT1___* LibSvcFloatVector;
	typedef struct ___LIBSVCAPIT2___* LibSvcDoubleDimsFloatVector;
	typedef struct ___LIBSVCAPIT3___* LibSvcInt16Vector;
	typedef struct ___LIBSVCAPIT4___* LibSvcUInt64Vector;
	typedef struct ___LIBSVCAPIT5___* LibSvcMelType;
	typedef struct ___LIBSVCAPIT6___* LibSvcSliceType;
	typedef struct ___LIBSVCAPIT7___* LibSvcSlicesType;
	typedef struct ___LIBSVCAPIT8___* LibSvcModel;
	typedef struct ___LIBSVCAPIT9___* LibSvcVocoderModel;
	typedef struct ___LIBSVCAPIT10___* LibSvcEnv;
	typedef const ___LIBSVCAPIT1___* LibSvcCFloatVector;
	typedef const ___LIBSVCAPIT2___* LibSvcCDoubleDimsFloatVector;
	typedef const ___LIBSVCAPIT3___* LibSvcCInt16Vector;
	typedef const ___LIBSVCAPIT4___* LibSvcCUInt64Vector;
	typedef const ___LIBSVCAPIT5___* LibSvcCMelType;
	typedef const ___LIBSVCAPIT6___* LibSvcCSliceType;
	typedef const ___LIBSVCAPIT7___* LibSvcCSlicesType;
	typedef const ___LIBSVCAPIT10___* LibSvcCEnv;

	enum LibSvcExecutionProviders
	{
		CPU = 0,
		CUDA = 1,
		DML = 2
	};

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

	struct LibSvcSlicerSettings
	{
		INT32 SamplingRate;
		double Threshold;
		double MinLength;
		INT32 WindowLength;
		INT32 HopSize;
	};

	struct LibSvcParams
	{
		float NoiseScale;								//噪声修正因子				[   0 ~ 10   ]
		INT64 Seed;										//种子						[   INT64    ]
		INT64 SpeakerId;								//默认角色ID					[   0 ~ NS   ]
		INT64 SpkCount;									//模型角色数					[	  NS     ]
		float IndexRate;								//索引比						[   0 ~ 1    ]
		float ClusterRate;								//聚类比						[   0 ~ 1    ]
		float DDSPNoiseScale;							//DDSP噪声修正因子			[   0 ~ 10   ]
		float Keys;										//升降调						[ -64 ~ 64   ]
		size_t MeanWindowLength;						//均值滤波器窗口大小			[   1 ~ 20   ]
		size_t Pndm;									//Diffusion加速倍数			[   1 ~ 200  ]
		size_t Step;									//Diffusion总步数			[   1 ~ 1000 ]
		float TBegin;									//Reflow起始点
		float TEnd;										//Reflow终止点
		LPWSTR Sampler;									//Diffusion采样器			["Pndm" "DDim"]
		LPWSTR ReflowSampler;							//Reflow采样器				["Eular" "Rk4" "Heun" "Pecece"]
		LPWSTR F0Method;								//F0提取算法					["Dio" "Harvest" "RMVPE" "FCPE"]
		LibSvcVocoderModel VocoderModel;								//声码器模型					Diffusion模型必须设定该项目
		INT32 VocoderHopSize;							//声码器HopSize				[    Hop     ]
		INT32 VocoderMelBins;							//声码器MelBins				[    Bins    ]
		INT32 VocoderSamplingRate;						//声码器采样率				[     SR     ]
		INT32 __DEBUG__MODE__;
	};

	struct DiffusionSvcPaths
	{
		LPWSTR Encoder;
		LPWSTR Denoise;
		LPWSTR Pred;
		LPWSTR After;
		LPWSTR Alpha;
		LPWSTR Naive;

		LPWSTR DiffSvc;
	};

	struct ReflowSvcPaths
	{
		LPWSTR Encoder;
		LPWSTR VelocityFn;
		LPWSTR After;
	};

	struct VitsSvcPaths
	{
		LPWSTR VitsSvc;
	};

	struct LibSvcClusterConfig
	{
		INT64 ClusterCenterSize;
		LPWSTR Path;
		LPWSTR Type; //"KMeans" "Index"
	};

	struct LibSvcHparams
	{
		LPWSTR TensorExtractor;
		LPWSTR HubertPath;
		DiffusionSvcPaths DiffusionSvc;
		VitsSvcPaths VitsSvc;
		ReflowSvcPaths ReflowSvc;
		LibSvcClusterConfig Cluster;

		INT32 SamplingRate;

		INT32 HopSize;
		INT64 HiddenUnitKDims;
		INT64 SpeakerCount;
		INT32 EnableCharaMix;
		INT32 EnableVolume;
		INT32 VaeMode;

		INT64 MelBins;
		INT64 Pndms;
		INT64 MaxStep;
		float SpecMin;
		float SpecMax;
		float Scale;
	};

#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	LibSvcApi void InitLibSvcHparams(
		LibSvcHparams* _Input
	);

	LibSvcApi void InitLibSvcParams(
		LibSvcParams* _Input
	);

	LibSvcApi void InitLibSvcSlicerSettings(
		LibSvcSlicerSettings* _Input
	);

	//FloatVector - vector<float>

	LibSvcApi float* LibSvcGetFloatVectorData(
		LibSvcFloatVector _Obj
	);

	LibSvcApi size_t LibSvcGetFloatVectorSize(
		LibSvcFloatVector _Obj
	);

	LibSvcApi LibSvcFloatVector LibSvcAllocateFloatVector();

	LibSvcApi void LibSvcReleaseFloatVector(
		LibSvcFloatVector _Obj
	);

	//DFloatVector - vector<vector<float>>

	LibSvcApi LibSvcFloatVector LibSvcGetDFloatVectorData(
		LibSvcDoubleDimsFloatVector _Obj,
		size_t _Index
	);

	LibSvcApi size_t LibSvcGetDFloatVectorSize(
		LibSvcDoubleDimsFloatVector _Obj
	);

	//Int16Vector - vector<int16_t>

	LibSvcApi LibSvcInt16Vector LibSvcAllocateInt16Vector();

	LibSvcApi void LibSvcReleaseInt16Vector(
		LibSvcInt16Vector _Obj
	);

	LibSvcApi void LibSvcSetInt16VectorLength(
		LibSvcInt16Vector _Obj,
		size_t _Size
	);

	LibSvcApi void LibSvcInsertInt16Vector(
		LibSvcInt16Vector _ObjA,
		LibSvcInt16Vector _ObjB
	);

	LibSvcApi short* LibSvcGetInt16VectorData(
		LibSvcInt16Vector _Obj
	);

	LibSvcApi size_t LibSvcGetInt16VectorSize(
		LibSvcInt16Vector _Obj
	);

	//UInt64Vector - vector<size_t>

	LibSvcApi LibSvcUInt64Vector LibSvcAllocateUInt64Vector();

	LibSvcApi void LibSvcReleaseUInt64Vector(
		LibSvcUInt64Vector _Obj
	);

	LibSvcApi void LibSvcSetUInt64VectorLength(
		LibSvcUInt64Vector _Obj,
		size_t _Size
	);

	LibSvcApi size_t* LibSvcGetUInt64VectorData(
		LibSvcUInt64Vector _Obj
	);

	LibSvcApi size_t LibSvcGetUInt64VectorSize(
		LibSvcUInt64Vector _Obj
	);

	//Mel - pair<vector<float>, int64_t>

	LibSvcApi LibSvcMelType LibSvcAllocateMel();

	LibSvcApi void LibSvcReleaseMel(
		LibSvcMelType _Obj
	);

	LibSvcApi LibSvcFloatVector LibSvcGetMelData(
		LibSvcMelType _Obj
	);

	LibSvcApi INT64 LibSvcGetMelSize(
		LibSvcMelType _Obj
	);

	//Slice - MoeVoiceStudioSvcSlice

	LibSvcApi LibSvcFloatVector LibSvcGetAudio(
		LibSvcSliceType _Obj
	);

	LibSvcApi LibSvcFloatVector LibSvcGetF0(
		LibSvcSliceType _Obj
	);

	LibSvcApi LibSvcFloatVector LibSvcGetVolume(
		LibSvcSliceType _Obj
	);

	LibSvcApi LibSvcDoubleDimsFloatVector LibSvcGetSpeaker(
		LibSvcSliceType _Obj
	);

	LibSvcApi UINT64 LibSvcGetSrcLength(
		LibSvcSliceType _Obj
	);

	LibSvcApi INT32 LibSvcGetIsNotMute(
		LibSvcSliceType _Obj
	);

	LibSvcApi void LibSvcSetSpeakerMixDataSize(
		LibSvcSliceType _Obj,
		size_t _NSpeaker
	);

	//Array Of Slice - MoeVoiceStudioSvcSlice

	LibSvcApi LibSvcSlicesType LibSvcAllocateSliceData();

	LibSvcApi void LibSvcReleaseSliceData(
		LibSvcSlicesType _Obj
	);

	LibSvcApi BSTR LibSvcGetAudioPath(
		LibSvcSlicesType _Obj
	);

	LibSvcApi LibSvcSliceType LibSvcGetSlice(
		LibSvcSlicesType _Obj,
		size_t _Index
	);

	LibSvcApi size_t LibSvcGetSliceCount(
		LibSvcSlicesType _Obj
	);

	/******************************************Fun**********************************************/

	LibSvcApi void LibSvcSetGlobalEnvDir(
		LPWSTR _Dir
	);

	LibSvcApi void LibSvcInit();

	LibSvcApi void LibSvcFreeString(
		BSTR _String
	);

	LibSvcApi LibSvcEnv LibSvcCreateEnv(
		UINT32 ThreadCount,
		UINT32 DeviceID,
		UINT32 Provider
	);

	LibSvcApi void LibSvcDestoryEnv(
		LibSvcEnv Env
	);

	LibSvcApi INT32 LibSvcSliceAudioI64(
		LibSvcCInt16Vector _Audio,
		const LibSvcSlicerSettings* _Setting,
		LibSvcUInt64Vector _Output
	);

	LibSvcApi INT32 LibSvcSliceAudio(
		LibSvcCFloatVector _Audio,
		const LibSvcSlicerSettings* _Setting,
		LibSvcUInt64Vector _Output
	);

	LibSvcApi INT32 LibSvcPreprocessI64(
		LibSvcCInt16Vector _Audio,
		LibSvcCUInt64Vector _SlicePos,
		INT32 _SamplingRate,
		INT32 _HopSize,
		double _Threshold,
		const wchar_t* _F0Method,
		LibSvcSlicesType _Output
	);

	LibSvcApi INT32 LibSvcPreprocess(
		LibSvcCFloatVector _Audio,
		LibSvcCUInt64Vector _SlicePos,
		INT32 _SamplingRate,
		INT32 _HopSize,
		double _Threshold,
		const wchar_t* _F0Method,
		LibSvcSlicesType _Output
	);

	LibSvcApi INT32 LibSvcStftI64(
		LibSvcCInt16Vector _Audio,
		INT32 _SamplingRate,
		INT32 _Hopsize,
		INT32 _MelBins,
		LibSvcMelType _Output
	);

	LibSvcApi INT32 LibSvcStft(
		LibSvcCFloatVector _Audio,
		INT32 _SamplingRate,
		INT32 _Hopsize,
		INT32 _MelBins,
		LibSvcMelType _Output
	);

	LibSvcApi INT32 LibSvcInferSlice(
		LibSvcModel _Model,
		LibSvcCSliceType _Slice,
		const LibSvcParams* _InferParams,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	LibSvcApi INT32 LibSvcInferAudio(
		LibSvcModel _Model,
		LibSvcSlicesType _Audio,
		const LibSvcParams* _InferParams,
		UINT64 _SrcLength,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	LibSvcApi INT32 LibSvcInferPCMData(
		LibSvcModel _Model,
		LibSvcCFloatVector _PCMData,
		const LibSvcParams* _InferParams,
		INT32 SamplingRate,
		LibSvcFloatVector _Output
	);

	LibSvcApi INT32 LibSvcShallowDiffusionInference(
		LibSvcModel _Model,
		LibSvcCFloatVector _16KAudioHubert,
		LibSvcMelType _Mel,
		LibSvcCFloatVector _SrcF0,
		LibSvcCFloatVector _SrcVolume,
		LibSvcCDoubleDimsFloatVector _SrcSpeakerMap,
		INT64 _SrcSize,
		const LibSvcParams* _InferParams,
		size_t* _Process,
		LibSvcFloatVector _Output
	);

	LibSvcApi INT32 LibSvcVocoderEnhance(
		LibSvcVocoderModel _Model,
		LibSvcEnv _Env,
		LibSvcMelType _Mel,
		LibSvcCFloatVector _F0,
		INT32 _VocoderMelBins,
		LibSvcFloatVector _Output
	);

	LibSvcApi LibSvcModel LibSvcLoadModel(
		UINT32 _T,
		const LibSvcHparams* _Config,
		ProgCallback _ProgressCallback,
		UINT32 _ExecutionProvider,
		UINT32 _DeviceID,
		UINT32 _ThreadCount
	);

	LibSvcApi INT32 LibSvcUnloadModel(
		LibSvcModel _Model
	);

	LibSvcApi LibSvcVocoderModel LibSvcLoadVocoder(
		LPWSTR VocoderPath,
		LibSvcEnv _Env
	);

	LibSvcApi INT32 LibSvcUnloadVocoder(
		LPWSTR VocoderPath,
		LibSvcEnv _Env
	);

	LibSvcApi void LibSvcClearCachedModel();

	LibSvcApi INT32 LibSvcReadAudio(
		LPWSTR _AudioPath,
		INT32 _SamplingRate,
		LibSvcFloatVector _Output
	);

	LibSvcApi void LibSvcWriteAudioFile(
		LibSvcFloatVector _PCMData,
		LPWSTR _OutputPath,
		INT32 _SamplingRate
	);

#ifdef __cplusplus
}
#endif