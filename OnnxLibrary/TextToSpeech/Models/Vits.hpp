
#pragma once
#include "OnnxLibrary/TextToSpeech/Util/Text2Speech.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

namespace Vits
{
    /**
     * @class SpeakerEmbedding
     * @brief Speaker embedding
     *
     * Following model paths are required:
     * - "Embedding"
     *
     * Following parameters are required:
     * - "GinChannel" - Int64 (default: 256)
     * - "SpeakerCount" - Int64 (default: 1)
     */
    class SpeakerEmbedding : public OnnxModelBase<SpeakerEmbedding>
    {
    public:
        SpeakerEmbedding() = delete;
        SpeakerEmbedding(
            const OnnxRuntimeEnvironment& _Environment,
            const HParams& Params,
            const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
        );
        ~SpeakerEmbedding() = default;

        SpeakerEmbedding(const SpeakerEmbedding&) = default;
        SpeakerEmbedding& operator=(const SpeakerEmbedding&) = default;
        SpeakerEmbedding(SpeakerEmbedding&&) noexcept = default;
        SpeakerEmbedding& operator=(SpeakerEmbedding&&) noexcept = default;

        Tensor<Float, 2, Device::CPU> Forward(const Tensor<Float, 2, Device::CPU>& Input) const;

    private:
        Int64 _MyGinChannel = 256;
        Int64 _MyEmbeddingCount = 1;
        Tensor<Float, 2, Device::CPU> _MyEmbedding;
    };

    /**
     * @class Encoder
     * @brief Encoder
     *
     * Following model paths are required:
     * - "Encoder"
     *
     * Following parameters are required:
     * - "HasLength" - bool (default: true)
     * - "HasEmotion" - bool (default: false)
     * - "HasTone" - bool (default: false)
     * - "HasLanguage" - bool (default: false)
     * - "HasBert" - bool (default: false)
     * - "HasClap" - bool (default: false)
     * - "HasSpeaker" - bool (default: false)
     * - "EncoderSpeaker" - bool (default: false)
     * - "HasVQ" - bool (default: false)
     * - "EmotionDims" - Int64 (default: 1024)
     * - "BertDims" - Int64 (default: 2048)
     * - "ClapDims" - Int64 (default: 512)
     * - "GinChannel" - Int64 (default: 256)
     * - "BertCount" - Int64 (default: 3)
     * - "VQCodebookSize" - Int64 (default: 10)
     * - "SpeakerCount" - Int64 (default: 1)
     *
     */
    class Encoder : public OnnxModelBase<Encoder>
    {
    public:
        struct Encoded
        {
            Tensor<Float, 3, Device::CPU> X;
            Tensor<Float, 3, Device::CPU> M_p;
            Tensor<Float, 3, Device::CPU> Logs_p;
            Tensor<Float, 3, Device::CPU> X_mask;
        };

        Encoder() = delete;
        Encoder(
            const OnnxRuntimeEnvironment& _Environment,
            const HParams& Params,
            const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
        );
        ~Encoder() = default;

        Encoder(const Encoder&) = default;
        Encoder& operator=(const Encoder&) = default;
        Encoder(Encoder&&) noexcept = default;
        Encoder& operator=(Encoder&&) noexcept = default;

        Encoded Forward(
            const Tensor<Int64, 2, Device::CPU>& PhonemeIds,
            const std::optional<const Tensor<Float, 2, Device::CPU>>& SpeakerEmbedding = std::nullopt,
            const std::optional<const Tensor<Float, 2, Device::CPU>>& Emotion = std::nullopt,
            const std::optional<const Tensor<Int64, 2, Device::CPU>>& ToneIds = std::nullopt,
            const std::optional<const Tensor<Int64, 2, Device::CPU>>& LanguageIds = std::nullopt,
            const std::optional<const Tensor<Float, 4, Device::CPU>>& Bert = std::nullopt,
            const std::optional<Tensor<Float, 2, Device::CPU>>& Clap = std::nullopt,
            Int64 VQIndex = 0,
            Int64 SpeakerIndex = 0
        ) const;

    private:
        Int64 _MyEmotionDims = 1024;
        Int64 _MyBertDims = 2048;
        Int64 _MyClapDims = 512;
        Int64 _MyGinChannel = 256;
        Int64 _MyBertCount = 3;
        Int64 _MyVQCodebookSize = 10;
        Int64 _MySpeakerCount = 1;
        bool _HasLength = true;
        bool _HasEmotion = false;
        bool _HasTone = false;
        bool _HasLanguage = false;
        bool _HasBert = false;
        bool _HasClap = false;
        bool _HasSpeaker = false;
        bool _HasVQ = false;
    };

    /**
     * @class DurationPredictor
     * @brief Duration predictor
     *
     * Following model paths are required: (all optional)
     * - "DP"
     * - "SDP"
     *
     * Following parameters are required:
     * - "HasSpeaker" - bool (default: false)
     * - "GinChannel" - Int64 (default: 256)
     * - "ZinDims" - Int64 (default: 2)
     */
    class DurationPredictor
    {
    public:
        DurationPredictor() = delete;
        DurationPredictor(
            const OnnxRuntimeEnvironment& _Environment,
            const HParams& Params,
            const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
        );
        ~DurationPredictor() = default;

        DurationPredictor(const DurationPredictor&) = default;
        DurationPredictor& operator=(const DurationPredictor&) = default;
        DurationPredictor(DurationPredictor&&) noexcept = default;
        DurationPredictor& operator=(DurationPredictor&&) noexcept = default;

        Tensor<Float32, 3, Device::CPU> Forward(
            const Tensor<Float32, 3, Device::CPU>& X,
            const Tensor<Float32, 3, Device::CPU>& X_Mask,
            const std::optional<const Tensor<Float32, 2, Device::CPU>>& SpeakerEmbedding = std::nullopt,
            float DurationPredictorNoiseScale = 0.8f,
            float SdpRatio = 1.0f,
            Int64 Seed = 114514
        ) const;

    private:
        OnnxModelBase<void> _MyDP;
        OnnxModelBase<void> _MySDP;
        Int64 _MyGinChannel = 256;
        Int64 _MyZinDims = 2;
        bool _HasSpeaker = false;

    public:
        operator bool() const noexcept { return _MyDP || _MySDP; }

        void SetTerminate() const
        {
            if (_MyDP)
                _MyDP.SetTerminate();
            if (_MySDP)
                _MySDP.SetTerminate();
        }

        void UnTerminate() const
        {
            if (_MyDP)
                _MyDP.UnTerminate();
            if (_MySDP)
                _MySDP.UnTerminate();
        }

    protected:
        ONNXTensorElementDataType GetInputType(size_t Index) const
        {
            if (Index == 114)
                return _MyDP ? _MyDP.GetInputTypes().Back() : _MySDP.GetInputTypes().Back();
            return _MyDP ? _MyDP.GetInputTypes()[Index] : _MySDP.GetInputTypes()[Index];
        }
        const TemplateLibrary::Vector<Int64>& GetInputDims(size_t Index) const
        {
            if (Index == 114)
                return _MyDP ? _MyDP.GetInputDims().Back() : _MySDP.GetInputDims().Back();
            return _MyDP ? _MyDP.GetInputDims()[Index] : _MySDP.GetInputDims()[Index];
        }
        DLogger GetLogger() const
        {
            return _MyDP ? _MyDP.GetLoggerPtr() : _MySDP.GetLoggerPtr();
        }
        Ort::MemoryInfo* GetMemoryInfo() const
        {
            return _MyDP ? _MyDP.GetMemoryInfo() : _MySDP.GetMemoryInfo();
        }
    };
}

/*class Vits : public TextToSpeech
{
public:
    Vits(
        const ModelHParams& _Config,
        const ProgressCallback& _ProgressCallback,
        const DurationCallback& _DurationCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

    ~Vits() override = default;

    Vits(const Vits&) = default;
    Vits& operator=(const Vits&) = default;
    Vits(Vits&&) noexcept = default;
    Vits& operator=(Vits&&) noexcept = default;

    DragonianLibSTL::Vector<float> Inference(TTSInputData& InputData, const TTSParams& Params) const override;

protected:

    std::string VitsType;
    int64_t DefBertSize = 1024;
    int64_t VQCodeBookSize = 10;
    int64_t BertCount = 3;

    bool UseTone = false;
    bool UseBert = false;
    bool UseLength = false;
    bool UseLanguage = false;
    bool EncoderG = false;
    bool ReferenceBert = false;
    bool UseVQ = false;
    bool UseClap = false;

    std::unordered_map<std::wstring, size_t> Emotion2Id;
    bool Emotion = false;
    EmotionLoader EmotionVector;

    std::shared_ptr<Ort::Session> sessionEnc_p = nullptr;
    std::shared_ptr<Ort::Session> sessionEmb = nullptr;
    std::shared_ptr<Ort::Session> sessionSdp = nullptr;
    std::shared_ptr<Ort::Session> sessionDp = nullptr;
    std::shared_ptr<Ort::Session> sessionFlow = nullptr;
    std::shared_ptr<Ort::Session> sessionDec = nullptr;

    std::vector<const char*> EncoderInputNames = { "x" };
    static inline const std::vector<const char*> EncoderOutputNames = { "xout", "m_p", "logs_p", "x_mask" };

    std::vector<const char*> SdpInputNames = { "x", "x_mask", "zin" };
    static inline const std::vector<const char*> SdpOutputNames = { "logw" };

    std::vector<const char*> DpInputNames = { "x", "x_mask" };
    static inline const std::vector<const char*> DpOutputNames = { "logw" };

    std::vector<const char*> FlowInputNames = { "z_p", "y_mask" };
    static inline const std::vector<const char*> FlowOutputNames = { "z" };

    std::vector<const char*> DecInputNames = { "z_in" };
    static inline const std::vector<const char*> DecOutputNames = { "o" };

    static inline const std::vector<const char*> EmbiddingInputNames = { "sid" };
    static inline const std::vector<const char*> EmbiddingOutputNames = { "g" };

    static inline const std::vector<const char*> VistBertInputNames =
    { "bert_0", "bert_1", "bert_2", "bert_3", "bert_4", "bert_5", "bert_6", "bert_7", "bert_8", "bert_9", "bert_10", "bert_11", "bert_12", "bert_13", "bert_14", "bert_15", "bert_16", "bert_17", "bert_18", "bert_19", "bert_20", "bert_21", "bert_22", "bert_23", "bert_24", "bert_25", "bert_26", "bert_27", "bert_28", "bert_29", "bert_30", "bert_31", "bert_32", "bert_33", "bert_34", "bert_35", "bert_36", "bert_37", "bert_38", "bert_39", "bert_40", "bert_41", "bert_42", "bert_43", "bert_44", "bert_45", "bert_46", "bert_47", "bert_48", "bert_49", "bert_50", "bert_51", "bert_52", "bert_53", "bert_54", "bert_55", "bert_56", "bert_57", "bert_58", "bert_59", "bert_60", "bert_61", "bert_62", "bert_63", "bert_64", "bert_65", "bert_66", "bert_67", "bert_68", "bert_69", "bert_70", "bert_71", "bert_72", "bert_73", "bert_74", "bert_75", "bert_76", "bert_77", "bert_78", "bert_79", "bert_80", "bert_81", "bert_82", "bert_83", "bert_84", "bert_85", "bert_86", "bert_87", "bert_88", "bert_89", "bert_90", "bert_91", "bert_92", "bert_93", "bert_94", "bert_95", "bert_96", "bert_97", "bert_98", "bert_99" };

private:
    DragonianLibSTL::Vector<float> GetEmotionVector(
        const DragonianLibSTL::Vector<std::wstring>& EmotionSymbol
    ) const;

    DragonianLibSTL::Vector<float> GetSpeakerEmbedding(
        const std::map<int64_t, float>& SpeakerMixIds
    ) const;
};*/

_D_Dragonian_Lib_Lib_Text_To_Speech_End