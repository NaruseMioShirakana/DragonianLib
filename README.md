# DragonianLib

A C++ library include tensor library, audio(1d-signal) library, text library, image library, video library, and so on.
A C++ library for singing voice conversion, text to speech, super resolution, music transcription, and so on.

## Bulid Options

- `DRAGONIANLIB_DML`: enable DML backend (default: OFF)
- `DRAGONIANLIB_CUDA`: enable CUDA backend (default: OFF)
- `DRAGONIANLIB_ROCM`: enable ROCM backend (default: OFF) TODO
- `DRAGONIANLIB_SHARED_LIBS`: build shared libs (default: OFF)
- `DRAGONIANLIB_STATIC_FFMPEG`: use static FFMPEG (default: ON)
- `DRAGONIANLIB_ORT_SUPER_RESOLUTION`: enable ONNX Runtime super resolution (default: ON)
- `DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION`: enable ONNX Runtime singing voice conversion (default: ON)
- `DRAGONIANLIB_ORT_TEXT_TO_SPEECH`: enable ONNX Runtime text to speech (default: ON)
- `DRAGONIANLIB_ORT_MUSIC_TRANSCRIPTION`: enable ONNX Runtime music transcription (default: ON)
- `DRAGONIANLIB_TRT_SUPER_RESOLUTION`: enable TensorRT super resolution (default: OFF)
- `DRAGONIANLIB_TRT_SINGING_VOICE_CONVERSION`: enable TensorRT singing voice conversion (default: OFF)
- `DRAGONIANLIB_TRT_TEXT_TO_SPEECH`: enable TensorRT text to speech (default: OFF)
- `DRAGONIANLIB_TRT_MUSIC_TRANSCRIPTION`: enable TensorRT music transcription (default: OFF)
- `DRAGONIANLIB_NCNN_SUPER_RESOLUTION`: enable NCNN super resolution (default: OFF)
- `DRAGONIANLIB_NCNN_SINGING_VOICE_CONVERSION`: enable NCNN singing voice conversion (default: OFF)
- `DRAGONIANLIB_NCNN_TEXT_TO_SPEECH`: enable NCNN text to speech (default: OFF)
- `DRAGONIANLIB_NCNN_MUSIC_TRANSCRIPTION`: enable NCNN music transcription (default: OFF)

## How to Build

1. clone full repo:
```bash
    git clone https://github.com/NaruseMioShirakana/DragonianLib.git
    cd DragonianLib
    git submodule update --init --recursive
```

2. create a build directory:
```bash
    mkdir build
    cd build
```

3. configure the build directory:
```bash
    cmake .. -DONNXRUNTIME_INCLUDE_DIRS=<ONNX Runtime headers> [required when ONNX is enabled]
             -DONNXRUNTIME_LIBRARIES=<ONNX Runtime lib> [required when ONNX is enabled]
             -DTENSORRT_INCLUDE_DIRS=<TensorRT headers> [required when TensorRT is enabled]
             -DTENSORRT_LIBRARIES=<TensorRT lib> [required when TensorRT is enabled]
             -DNCNN_INCLUDE_DIRS=<NCNN headers> [required when NCNN is enabled]
             -DNCNN_LIBRARIES=<NCNN lib> [required when NCNN is enabled]
             -DFFMPEG_INCLUDE_DIRS=<FFMPEG headers>
             -DFFMPEG_LIBRARIES=<FFMPEG lib>
             -DYYJSON_INCLUDE_DIRS=<YYJSON headers> [will be enabled if find_package failed]
             -DYYJSON_LIBRARIES=<YYJSON lib> [will be enabled if find_package failed]
             -DLIBREMIDI_INCLUDE_DIRS=<libremidi headers> [will be enabled if find_package failed]
             -DLIBREMIDI_LIBRARIES=<libremidi lib> [will be enabled if find_package failed]
             -DFFTW3_INCLUDE_DIRS=<FFTW headers> [will be enabled if find_package failed]
             -DFFTW3_LIBRARIES=<FFTW lib> [will be enabled if find_package failed]
             -DWORLD_INCLUDE_DIRS=<WORLD headers> [will be enabled if find_package failed]
             -DWORLD_LIBRARIES=<WORLD lib> [will be enabled if find_package failed]
             -DFAISS_INCLUDE_DIRS=<Faiss headers> [will be enabled if find_package failed]
             -DFAISS_LIBRARIES=<Faiss lib> [will be enabled if find_package failed]
             -DOpenBLAS_INCLUDE_DIRS=<OpenBLAS headers> [will be enabled if find_package failed]
             -DOpenBLAS_LIBRARIES=<OpenBLAS lib> [will be enabled if find_package failed]
```

4. build：
```bash
	cmake --build .
```

## References

>- [libcudacxx](https://developer.nvidia.com/cuda-toolkit) (enabled if `DRAGONIANLIB_CUDA` is set to `ON`))
>- [TensorRT](https://developer.nvidia.com/tensorrt) (enabled if `DRAGONIANLIB_TENSORRT` is set to `ON`)

>- [FFMPEG](https://ffmpeg.org/) (audio/video codec library)
>- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (Matrix library, used for some matrix operations)
>- [libremidi](https://github.com/celtera/libremidi) (a MIDI library, used for MIDI I/O)
>- [FFTW3](http://www.fftw.org/) (a fast DFT library)
>- [World](https://github.com/mmorise/World) (audio analysis library, used for extracting F0)
>- [Faiss](https://github.com/facebookresearch/faiss) (index cluster, used for units cluster)
>- [KDTree](https://github.com/crvs/KDTree) (an implementation of KDTree, used for units cluster)

>- [ONNXRuntime](https://onnxruntime.ai/) (enabled if `DRAGONIANLIB_ONNXRUNTIME` is set to `ON`)))
>- [NCNN](https://github.com/Tencent/ncnn) (enabled if `DRAGONIANLIB_NCNN` is set to `ON`)))

>- [pypinyin](https://github.com/mozillazg/python-pinyin) (a python library for Chinese G2P, DragonianLib implemented a C++ version)

>- [SoVits-SVC](https://github.com/svc-develop-team/so-vits-svc) (Officially maintained version of SoVits-SVC)
>- [Origin-SoVits-SVC](https://github.com/innnky/so-vits-svc) (original version of SoVits-SVC)
>- [Diffusion-SVC & Reflow-SVC](https://github.com/CNChTu/Diffusion-SVC) (voice conversion model based on diffusion or reflow)
>- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) (voice conversion model based on DDSP and Reflow)
>- [RVC](https://github.com/rvc-project/Retrieval-based-Voice-Conversion-WebUI) (voice conversion model based on vits)

>- [Vits](https://github.com/jaywalnut310/VITS) (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)
>- [EmotionalVits](https://github.com/innnky/emotional-vits) (Vits with emotional control)
>- [Vits2](https://github.com/daniilrobnikov/vits2) (Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design)
>- [Bert-Vits2](https://github.com/fishaudio/Bert-Vits2) (VITS2 Backbone with multilingual bert)
>- [Gpt-SoVits](https://github.com/RVC-Boss/Gpt-SoVits) (A Powerful Few-shot Voice Conversion and Text-to-Speech)
>- [FishSpeech](https://fish.audio/) (Sota open-source TTS)

>- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (a Super-Resolution Network)
>- [Real-HATGAN](https://github.com/TeamMoeAI/MoeSR/tree/main) (a Super-Resolution Network)

>- [PianoTranscription](https://github.com/bytedance/piano_transcription) (a piano transcription model, used for converting audio to MIDI)

## Output Directories

> if you set `DRAGONIANLIB_SHARED_LIBS` to `ON`, the output directories will be:
> - `OutPuts/Shared/RelWithDebInfo`
> - `OutPuts/Shared/Release`
> - `OutPuts/Shared/Debug`

> if you set `DRAGONIANLIB_SHARED_LIBS` to `OFF`, the output directories will be:
> - `OutPuts/Static/RelWithDebInfo`
> - `OutPuts/Static/Release`
> - `OutPuts/Static/Debug`