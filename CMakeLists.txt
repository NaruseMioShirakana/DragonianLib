cmake_minimum_required (VERSION 3.13)

#Options
option(DRAGONIANLIB_DML                     "dml backend"         OFF)
option(DRAGONIANLIB_CUDA                    "cuda backend"        OFF)
option(DRAGONIANLIB_ROCM                    "rocm backend"        OFF)
option(DRAGONIANLIB_SHARED_LIBS             "build shared libs"   OFF)
option(DRAGONIANLIB_ONNXRUNTIME             "build ort libs"      ON)
option(DRAGONIANLIB_TENSORRT                "build trt libs"      OFF)
option(DRAGONIANLIB_STATIC_FFMPEG           "use static ffmpeg"   ON)
option(DRAGONIANLIB_BUILD_DEMO              "build demo"          ON)
add_definitions(-DDRAGONIANLIB_ALLOC_ALIG=32)
add_definitions(-DDRAGONIANLIB_ALIG_DIM_SHAPE=8)  #Don't Change This Value'
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_FRONT=8)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_BACK=32)
add_definitions(-DDRAGONIANLIB_EMPTY_CAPACITY=16)
add_definitions(-DDRAGONIANLIB_PADDING_COUNT=64000)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE=65536)
add_definitions(-DDRAGONIANLIB_NAME_MAX_SIZE=1024)
add_definitions(-DUNICODE)
add_definitions(-D_UNICODE)
add_definitions(-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)
set(BUILD_TESTING OFF)
set(BUILD_BENCHMARKS OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_C_API ON)
set(DRAGONIANLIB_LANGUAGES CXX)
# set BLAS_LIBRARIES BLAS_INCLUDE_DIRS FFMPEG_LIBRARIES FFMPEG_INCLUDE_DIRS ONNXRT_LIBRARIES ONNXRT_INCLUDE
# set ONNXRT_DYNLIBS
enable_language(ASM_MASM)

#project name
set(DRAGONIANLIB_LIB DragonianLib)

#device
if(DRAGONIANLIB_ONNXRUNTIME)
    message("Onnx Build")
endif ()

if(DRAGONIANLIB_CUDA)
    add_definitions(-DDRAGONIANLIB_ENABLECUDA)
    set(FAISS_ENABLE_GPU ON)
    list(APPEND DRAGONIANLIB_LANGUAGES CUDA)
    set(CUDA_ARCHITECTURES all)
endif()
if (DRAGONIANLIB_ROCM)
    add_definitions(-DDRAGONIANLIB_ENABLEROCM)
    list(APPEND DRAGONIANLIB_LANGUAGES HIP)
endif ()
if (DRAGONIANLIB_DML)
    add_definitions(-DDRAGONIANLIB_ENABLEDML)
endif ()

#project
if(DRAGONIANLIB_ONNXRUNTIME)
    set(DRAGONIANLIB_SUFFIX "Ort")
else()
    set(DRAGONIANLIB_SUFFIX "Tensor")
endif()
set(DRAGONIANLIB_LIB ${DRAGONIANLIB_LIB}${DRAGONIANLIB_SUFFIX})
project (
    ${DRAGONIANLIB_LIB}
    VERSION 1.0.0
    LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

if(NOT DEFINED FFMPEG_LIBRARIES)
    set(FFMPEG_LIBRARIES)
    if(DRAGONIANLIB_STATIC_FFMPEG)
        list(
            APPEND FFMPEG_LIBRARIES
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavcodec.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavdevice.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavfilter.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavformat.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavutil.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libswresample.a
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libswscale.a
        )
        if(WIN32)
            list(
                APPEND FFMPEG_LIBRARIES
                dxva2.lib
                evr.lib
                mf.lib
                mfplat.lib
                mfplay.lib
                mfreadwrite.lib
                mfuuid.lib
                bcrypt.lib
                ws2_32.lib
                Secur32.lib
                Strmiids.lib
            )
        endif()
    else()
        list(
            APPEND FFMPEG_LIBRARIES
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avcodec.lib
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avformat.lib
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avutil.lib
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/swresample.lib
            ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/swscale.lib
        )
    endif()
endif()
if(NOT DEFINED FFMPEG_INCLUDE_DIRS)
    set(FFMPEG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/include)
endif()

if(NOT DEFINED BLAS_LIBRARIES)
    set(BLAS_LIBRARIES)
endif()
if(NOT DEFINED BLAS_INCLUDE_DIRS)
    set(BLAS_INCLUDE_DIRS)
    list(
        APPEND BLAS_INCLUDE_DIRS
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}
    )
endif()

#outdirs
if(DRAGONIANLIB_SHARED_LIBS)
    set(DRAGONIANLIB_TYPE_DIR "Shared")
else()
    set(DRAGONIANLIB_TYPE_DIR "Static")
endif()
if(DRAGONIANLIB_ONNXRUNTIME)
    set(DRAGONIANLIB_TYPE_ORT_DIR "OnnxRuntime")
else()
    set(DRAGONIANLIB_TYPE_ORT_DIR "TensorLib")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_ORT_DIR}/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_ORT_DIR}/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_ORT_DIR}/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Runtime)
set(LIBRARY_OUTPUT_PATH             ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_ORT_DIR}/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/OLibrary)

#bulid type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message("Build Release")
    add_compile_options(/GS- /arch:AVX2 /O2 /Ob2 /Ot)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    add_compile_options(/Zi /Gy /arch:AVX2 /O2 /Ob2 /Ot)
else()
    message("Build Debug")
    add_definitions(-DDRAGONIANLIB_DEBUG)
    add_compile_options(/Zi /Gy)
endif()

if(MSVC)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif()
endif()

set(DRAGONIANLIB_LIBRARIES_SRC)
set(DRAGONIANLIB_TENSORLIB_SRC)
set(DRAGONIANLIB_ORTLIB_SRC)
set(DRAGONIANLIB_TRTLIB_SRC)

file(GLOB_RECURSE DRAGONIANLIB_LIBRARIES_SRC 
    ${PROJECT_SOURCE_DIR}/Libraries/*.cpp
    ${PROJECT_SOURCE_DIR}/Libraries/*.h
    ${PROJECT_SOURCE_DIR}/Libraries/*.c
    ${PROJECT_SOURCE_DIR}/Libraries/*.hpp
    #${PROJECT_SOURCE_DIR}/Libraries/*.asm
)

file(GLOB_RECURSE DRAGONIANLIB_TENSORLIB_SRC
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.c
)

#src
if(DRAGONIANLIB_ONNXRUNTIME)
    file(GLOB_RECURSE DRAGONIANLIB_ORTLIB_SRC
        ${PROJECT_SOURCE_DIR}/OnnxLibrary/*.c
        ${PROJECT_SOURCE_DIR}/OnnxLibrary/*.cpp
        ${PROJECT_SOURCE_DIR}/OnnxLibrary/*.h
        ${PROJECT_SOURCE_DIR}/OnnxLibrary/*.hpp
    )
    if(DRAGONIANLIB_TENSORRT)
    file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_SRC
        ${PROJECT_SOURCE_DIR}/TensorRT/*.hpp
        ${PROJECT_SOURCE_DIR}/TensorRT/*.h
        ${PROJECT_SOURCE_DIR}/TensorRT/*.cpp
        ${PROJECT_SOURCE_DIR}/TensorRT/*.c
    )
    endif()
endif()

set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
#build shared lib?
if(DRAGONIANLIB_SHARED_LIBS)
    add_library(${DRAGONIANLIB_LIB} SHARED ${DRAGONIANLIB_LIBRARIES_SRC} ${DRAGONIANLIB_TENSORLIB_SRC} ${DRAGONIANLIB_ORTLIB_SRC} ${DRAGONIANLIB_TRTLIB_SRC})
    add_definitions(-DDRAGONIANLIB_EXPORT)
    target_compile_definitions(${DRAGONIANLIB_LIB} PRIVATE -DDRAGONIANLIB_EXPORT)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    add_library(${DRAGONIANLIB_LIB} STATIC ${DRAGONIANLIB_LIBRARIES_SRC} ${DRAGONIANLIB_TENSORLIB_SRC} ${DRAGONIANLIB_ORTLIB_SRC} ${DRAGONIANLIB_TRTLIB_SRC})
endif()

#link
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/Third-Party-Libraries")
set(BUILD_TESTING OFF)

find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
    set(OpenBLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/lib/openblas.lib)
    set(OpenBLAS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/include/openblas)
endif()
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${OpenBLAS_LIBRARIES})
include_directories(${OpenBLAS_INCLUDE_DIRS})
set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
set(LAPACK_LIBRARIES ${OpenBLAS_LIBRARIES})

target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FFMPEG_LIBRARIES})
include_directories(${FFMPEG_INCLUDE_DIRS})

add_subdirectory(Third-Party-Libraries/World)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC world)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC world_tool)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/src)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/src/world)

set(BUILD_TESTS OFF)
set(ENABLE_THREADS ON)
set(ENABLE_AVX2 ON)
set(WITH_COMBINED_THREADS ON)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(Third-Party-Libraries/fftw-3.3.10)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC fftw3)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/fftw-3.3.10/api)
unset(BUILD_TESTS)

set(FAISS_OPT_LEVEL "avx2")
add_subdirectory(Third-Party-Libraries/Faiss)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC faiss)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/Faiss/faiss)

add_subdirectory(Third-Party-Libraries/yyjson)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC yyjson)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/yyjson/src)

set(LIBREMIDI_NI_MIDI2 OFF)
add_subdirectory(Third-Party-Libraries/libmidi)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC libremidi)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/include)

#if(NOT ONNXRT_LIBRARIES)
#    set(ONNXRT_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/onnxruntime/runtimes/win-x64/native/onnxruntime.lib)
#endif()
#if(NOT ONNXRT_INCLUDE)
#    set(ONNXRT_INCLUDE ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/onnxruntime/build/native/include)
#endif()

#.\build.bat --use_cuda --parallel --use_dml --config Release --skip_tests --cuda_home "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5" --cudnn_home "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
if(DRAGONIANLIB_ONNXRUNTIME)
    include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core)
    include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core/session)
    set(ONNXRUNTIME_LIBRARIES 
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_shared.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_dml.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_cuda.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime.lib
    )
    add_definitions(-DDRAGONIANLIB_ONNXRT_LIB)
    target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${ONNXRUNTIME_LIBRARIES})
else()
    
endif()

#project include
include_directories(${PROJECT_SOURCE_DIR}/include/Base)
if(DRAGONIANLIB_ONNXRUNTIME)
    include_directories(${PROJECT_SOURCE_DIR}/OnnxLibrary)
endif()
include_directories(${PROJECT_SOURCE_DIR}/Libraries)


if(DRAGONIANLIB_TENSORRT AND DRAGONIANLIB_ONNXRUNTIME)
    include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/include)
    target_link_libraries(
        ${DRAGONIANLIB_LIB} PUBLIC 
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_10.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_dispatch_10.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_lean_10.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_plugin_10.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_vc_plugin_10.lib
        ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvonnxparser_10.lib
    )
    find_package(libcudacxx REQUIRED)
    include_directories(${_libcudacxx_INCLUDE_DIR})
    target_link_libraries(
        ${DRAGONIANLIB_LIB} PUBLIC 
        libcudacxx::libcudacxx
    )
endif()

#standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET ${DRAGONIANLIB_LIB} PROPERTY CXX_STANDARD 23)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
    target_link_options(${DRAGONIANLIB_LIB} PRIVATE 
        /NATVIS:${PROJECT_SOURCE_DIR}/NatvisFile.natvis
    )
endif()

#demo
if (DRAGONIANLIB_BUILD_DEMO)
    if(DRAGONIANLIB_ONNXRUNTIME)
        set(DRAGONIANLIBORTDEMO on)
    endif()
    add_subdirectory(Demo)
endif()