# DragonianLib

A C++ library include tensor library, audio(1d-signal) library, text library, image library, video library, and so on.

## 构建选项

在构建 DragonianLib 时，可以通过以下选项启用或禁用特定的功能：

- `DRAGONIANLIB_DML`：启用 DML 后端（默认：OFF）
- `DRAGONIANLIB_CUDA`：启用 CUDA 后端（默认：OFF）
- `DRAGONIANLIB_ROCM`：启用 ROCM 后端（默认：OFF）TODO
- `DRAGONIANLIB_SHARED_LIBS`：构建共享库（默认：ON）
- `DRAGONIANLIB_ONNXRUNTIME`：构建 ONNX Runtime 库（默认：ON）
- `DRAGONIANLIB_TENSORRT`：构建 TensorRT 库（默认：OFF）
- `DRAGONIANLIB_BUILD_DEMO`：构建示例（默认：ON）

## 构建步骤

1. 克隆项目到本地：
```bash
    git clone https://github.com/NaruseMioShirakana/DragonianLib.git
    cd DragonianLib
```

2. 创建构建目录：
```bash
    mkdir build
    cd build
```

3. 运行 CMake 配置：
```bash
    cmake .. -DONNXRUNTIME_INCLUDE_DIRS=<ONNX Runtime 头文件目录> [启用Onnx时必须]
             -DONNXRUNTIME_LIBRARIES=<ONNX Runtime 库文件目录> [启用Onnx时必须]
             -DTENSORRT_INCLUDE_DIRS=<TensorRT 头文件目录> [启用TensorRT时必须]
             -DTENSORRT_LIBRARIES=<TensorRT 库文件目录> [启用TensorRT时必须]
             -DFFMPEG_INCLUDE_DIRS=<FFMPEG 头文件目录>
             -DFFMPEG_LIBRARIES=<FFMPEG 库文件目录>
             -DLIBREMIDI_INCLUDE_DIRS=<libremidi 头文件目录> [当find_package失败时启用]
             -DLIBREMIDI_LIBRARIES=<libremidi 库文件目录> [当find_package失败时启用]
             -DFFTW3_INCLUDE_DIRS=<FFTW 头文件目录> [当find_package失败时启用]
             -DFFTW3_LIBRARIES=<FFTW 库文件目录> [当find_package失败时启用]
             -DWORLD_INCLUDE_DIRS=<WORLD 头文件目录> [当find_package失败时启用]
             -DWORLD_LIBRARIES=<WORLD 库文件目录> [当find_package失败时启用]
             -DFAISS_INCLUDE_DIRS=<Faiss 头文件目录> [当find_package失败时启用]
             -DFAISS_LIBRARIES=<Faiss 库文件目录> [当find_package失败时启用]
             -DOpenBLAS_INCLUDE_DIRS=<OpenBLAS 头文件目录> [当find_package失败时启用]
             -DOpenBLAS_LIBRARIES=<OpenBLAS 库文件目录> [当find_package失败时启用]
```

4. 构建项目：
```bash
	cmake --build .
```

## 依赖项

DragonianLib 依赖以下第三方库：

- FFMPEG
- OpenBLAS
- libcudacxx（如果启用 CUDA）
- libremidi
- FFTW3
- World
- Faiss
- ONNX Runtime（如果启用Onnx）
- TensorRT（如果启用TensorRT）
- KDTree
- pypinyin

这些库的路径和包含目录在 `CMakeLists.txt` 文件中进行了配置。

## 输出目录

构建输出将根据库类型（共享或静态）和构建类型（Release 或 Debug）放置在以下目录中：

- `OutPuts/Shared/Release`
- `OutPuts/Shared/Debug`
- `OutPuts/Static/Release`
- `OutPuts/Static/Debug`

## 示例

如果启用了 `DRAGONIANLIB_BUILD_DEMO` 选项，将会构建示例项目。示例项目的源代码位于 `Demo` 目录中。
