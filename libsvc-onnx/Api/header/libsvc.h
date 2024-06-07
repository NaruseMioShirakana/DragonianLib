#pragma once
#include "../../Modules/header/Modules.hpp"
#include <any>

namespace libsvccore
{
	using Config = MoeVoiceStudioCore::Hparams;
	using VitsSvc = MoeVoiceStudioCore::VitsSvc;
	using DiffusionSvc = MoeVoiceStudioCore::DiffusionSvc;
	using ReflowSvc = MoeVoiceStudioCore::ReflowSvc;
	using ClusterBase = MoeVoiceStudioCluster::MoeVoiceStudioBaseCluster;
	using TensorExtractorBase = MoeVSTensorPreprocess::MoeVoiceStudioTensorExtractor;
	using ProgressCallback = MoeVoiceStudioCore::MoeVoiceStudioModule::ProgressCallback;
	using ExecutionProvider = MoeVoiceStudioCore::MoeVoiceStudioModule::ExecutionProviders;
	using Slices = MoeVSProjectSpace::MoeVoiceStudioSvcData;
	using SingleSlice = MoeVSProjectSpace::MoeVoiceStudioSvcSlice;
	using Params = MoeVSProjectSpace::MoeVSSvcParams;
	enum class ModelType { Vits, Diffusion, Reflow };

	LibSvcApi void SliceAudio(size_t& _Id, const std::vector<int16_t>& _Audio, const InferTools::SlicerSettings& _Setting);

	LibSvcApi void Preprocess(size_t& _Id, const std::vector<int16_t>& _Audio, const std::vector<size_t>& _SlicePos, const InferTools::SlicerSettings& _Setting, int _SamplingRate, int _HopSize, const std::wstring& _F0Method);

	LibSvcApi int InferSlice(size_t& _Id, ModelType _T, const std::wstring& _Name, const SingleSlice& _Slice, const Params& _InferParams, size_t& _Process);

	LibSvcApi int ShallowDiffusionInference(size_t& _Id, const std::wstring& _Name, const std::vector<float>& _16KAudioHubert, const MoeVSProjectSpace::MoeVSSvcParams& _InferParams, const std::pair<std::vector<float>, int64_t>& _Mel, const std::vector<float>& _SrcF0, const std::vector<float>& _SrcVolume, const std::vector<std::vector<float>>& _SrcSpeakerMap, size_t& Process, int64_t SrcSize);
	
	LibSvcApi int Stft(size_t& _Id, const std::vector<double>& _NormalizedAudio, int _SamplingRate, int _Hopsize, int _MelBins);

	LibSvcApi int VocoderEnhance(size_t& _Id, const std::vector<float>& Mel, const std::vector<float>& F0, size_t MelSize, long VocoderMelBins);

	LibSvcApi int LoadModel(ModelType _T, const Config& _Config, const std::wstring& _Name, const ProgressCallback& _ProgressCallback,
		ExecutionProvider ExecutionProvider_ = ExecutionProvider::CPU,
		unsigned DeviceID_ = 0, unsigned ThreadCount_ = 0);

	LibSvcApi void UnloadModel(ModelType _T, const std::wstring& _Name);

	LibSvcApi std::any& GetData(size_t _Id);

	LibSvcApi void PopData(size_t _Id);

	/// <summary>
	/// ����Error���е���󳤶ȣ�����Error��Ϣ��
	/// </summary>
	/// <param name="Count">����</param>
	/// <returns></returns>
	LibSvcApi void SetMaxErrorCount(size_t Count);

	/// <summary>
	/// ��ȡError���еĵ���Index��
	/// </summary>
	/// <param name="Index">��������</param>
	/// <returns></returns>
	LibSvcApi std::wstring& GetLastError(size_t Index);

	/// <summary>
	/// ���Stft����
	/// </summary>
	/// <returns></returns>
	LibSvcApi void EmptyStftCache();

	/// <summary>
	/// ����������ģ��
	/// </summary>
	/// <param name="VocoderPath">������ģ��·��</param>
	/// <returns></returns>
	LibSvcApi void LoadVocoder(const std::wstring& VocoderPath);

	/// <summary>
	/// ��ʼ��������ʹ��ǰ������ã�
	/// </summary>
	/// <returns></returns>
	LibSvcApi void Init();

	/// <summary>
	/// ����ȫ�ֻ�������ҪӰ��F0Predictor��
	/// </summary>
	/// <returns></returns>
	LibSvcApi void SetGlobalEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);
}

namespace libsvc
{
	using libsvccore::Config;
	using libsvccore::VitsSvc;
	using libsvccore::DiffusionSvc;
	using libsvccore::ClusterBase;
	using libsvccore::TensorExtractorBase;
	using libsvccore::ProgressCallback;
	using libsvccore::ExecutionProvider;
	using libsvccore::ModelType;
	using libsvccore::Slices;
	using libsvccore::SingleSlice;
	using libsvccore::Params;
	using MelContainer = std::pair<std::vector<float>, int64_t>;

	using libsvccore::SetMaxErrorCount;
	using libsvccore::GetLastError;
	using libsvccore::EmptyStftCache;
	using libsvccore::LoadVocoder;
	using libsvccore::Init;
	using libsvccore::SetGlobalEnv;

	/// <summary>
	/// ж��һ��ģ��
	/// </summary>
	/// <param name="_T">ģ�����</param>
	/// <param name="_Name">ģ��ID</param>
	/// <returns></returns>
	inline void UnloadModel(
		ModelType _T,
		const std::wstring& _Name
	)
	{
		libsvccore::UnloadModel(_T, _Name);
	}

	/// <summary>
	/// ����һ��ģ��
	/// </summary>
	/// <param name="_T">ģ�����</param>
	/// <param name="_Config">������Ŀ</param>
	/// <param name="_Name">ģ��ID</param>
	/// <param name="_ProgressCallback">�������ص�����</param>
	/// <param name="ExecutionProvider_">ExecutionProvider</param>
	/// <param name="DeviceID_">GPUID</param>
	/// <param name="ThreadCount_">�߳���</param>
	/// <returns>�ɹ����أ�0�� ʧ�ܣ���0</returns>
	inline int LoadModel(
		ModelType _T,
		const Config& _Config,
		const std::wstring& _Name,
		const ProgressCallback& _ProgressCallback,
		ExecutionProvider ExecutionProvider_ = ExecutionProvider::CPU,
		unsigned DeviceID_ = 0,
		unsigned ThreadCount_ = 0
	)
	{
		return libsvccore::LoadModel(_T, _Config, _Name, _ProgressCallback, ExecutionProvider_, DeviceID_, ThreadCount_);
	}

	/**
	 * \brief ��Ƭ��Ƶ
	 * \param _Audio ������Ƶ��������PCM-Signed-Int16 ��������
	 * \param _Setting ��Ƭ������
	 * \return ��ƬPos
	 */
	inline std::vector<size_t> SliceAudio(
		const std::vector<int16_t>& _Audio,
		const InferTools::SlicerSettings& _Setting
	)
	{
		size_t _Ptr = 0;
		libsvccore::SliceAudio(_Ptr, _Audio, _Setting);
		std::vector temp = std::any_cast<std::vector<size_t>>(libsvccore::GetData(_Ptr));
		libsvccore::PopData(_Ptr);
		return temp;
	}

	/**
	 * \brief Ԥ������Ƭ�õ���Ƶ
	 * \param _Audio ������Ƶ
	 * \param _SlicePos ��ƬPos
	 * \param _Setting ��Ƭ������
	 * \param _SamplingRate ������
	 * \param _HopSize Hopsize
	 * \param _F0Method F0�㷨
	 * \return Ԥ���������ݣ������Լ���һ���������ֱ���ͽ�����������
	 */
	inline Slices PreprocessSlices(
		const std::vector<int16_t>& _Audio,
		const std::vector<size_t>& _SlicePos,
		const InferTools::SlicerSettings& _Setting,
		int _SamplingRate = 48000,
		int _HopSize = 512,
		const std::wstring& _F0Method = L"Dio"
	)
	{
		size_t _Ptr = 0;
		libsvccore::Preprocess(_Ptr, _Audio, _SlicePos, _Setting, _SamplingRate, _HopSize, _F0Method);
		Slices temp = std::any_cast<Slices>(libsvccore::GetData(_Ptr));
		libsvccore::PopData(_Ptr);
		return temp;
	}

	/**
	 * \brief ��ʱ����Ҷ�任��Mel��
	 * \param _NormalizedAudio ��һ����ƵPCM����
	 * \param _SamplingRate ������
	 * \param _Hopsize HopSize
	 * \param _MelBins MelBins
	 * \return Mel
	 */
	inline MelContainer Stft(
		const std::vector<double>& _NormalizedAudio,
		int _SamplingRate,
		int _Hopsize,
		int _MelBins
	)
	{
		size_t _Ptr = 0;
		libsvccore::Stft(_Ptr, _NormalizedAudio, _SamplingRate, _Hopsize, _MelBins);
		MelContainer temp = std::any_cast<MelContainer>(libsvccore::GetData(_Ptr));
		libsvccore::PopData(_Ptr);
		return temp;
	}

	/**
	 * \brief ����һ����Ƭ
	 * \param _T ģ������
	 * \param _Name ģ��ID
	 * \param _Slice ��Ƭ����
	 * \param _InferParams �������
	 * \param _Process �ܽ���
	 * \return ��ƵPCM����
	 */
	inline std::vector<int16_t> InferSlice(
		ModelType _T,
		const std::wstring& _Name,
		const SingleSlice& _Slice,
		const Params& _InferParams,
		size_t& _Process
	)
	{
		size_t _Ptr = 0;
		if(!libsvccore::InferSlice(_Ptr, _T, _Name, _Slice, _InferParams, _Process))
		{
			std::vector<int16_t> temp = std::any_cast<std::vector<int16_t>>(libsvccore::GetData(_Ptr));
			libsvccore::PopData(_Ptr);
			return temp;
		}
		return {};
	}

	/**
	 * \brief ǳ��ɢ����
	 * \param _Name Diffusionģ��ID
	 * \param _16KAudioHubert 16000�����ʵ�ԭʼ��Ƶ����ǰ����Ƶ��
	 * \param _InferParams �������
	 * \param _Mel Mel
	 * \param _SrcF0 ��Ƶ����
	 * \param _SrcVolume ��������
	 * \param _SrcSpeakerMap ˵���˱�������
	 * \param Process ��������
	 * \param SrcSize ԭʼ��Ƶ���ȣ���Ƭ����Ƭ�����ĳ��ȣ�
	 * \return ��ƵPCM����
	 */
	inline std::vector<int16_t> ShallowDiffusionInference(
		const std::wstring& _Name,
		const std::vector<float>& _16KAudioHubert,
		const MoeVSProjectSpace::MoeVSSvcParams& _InferParams,
		const MelContainer& _Mel,
		const std::vector<float>& _SrcF0,
		const std::vector<float>& _SrcVolume,
		const std::vector<std::vector<float>>& _SrcSpeakerMap,
		size_t& Process,
		int64_t SrcSize
	)
	{
		size_t _Ptr = 0;
		if(!libsvccore::ShallowDiffusionInference(_Ptr, _Name, _16KAudioHubert, _InferParams, _Mel, _SrcF0, _SrcVolume, _SrcSpeakerMap, Process, SrcSize))
		{
			std::vector<int16_t> temp = std::any_cast<std::vector<int16_t>>(libsvccore::GetData(_Ptr));
			libsvccore::PopData(_Ptr);
			return temp;
		}
		return {};
	}

	/**
	 * \brief ��������ǿ������������
	 * \param Mel Mel
	 * \param F0 ��Ƶ����
	 * \param MelSize Mel��֡��
	 * \param VocoderMelBins ��������MelBins
	 * \return PCM����
	 */
	inline std::vector<int16_t> VocoderEnhance(const std::vector<float>& Mel, const std::vector<float>& F0, size_t MelSize, long VocoderMelBins)
	{
		size_t _Ptr = 0;
		if(!libsvccore::VocoderEnhance(_Ptr, Mel, F0, MelSize, VocoderMelBins))
		{
			std::vector<int16_t> temp = std::any_cast<std::vector<int16_t>>(libsvccore::GetData(_Ptr));
			libsvccore::PopData(_Ptr);
			return temp;
		}
		return {};
	}
}
