cmake_minimum_required (VERSION 3.13)

#Options
option(DRAGONIANLIB_DML                     "dml backend"         OFF)
option(DRAGONIANLIB_CUDA                    "cuda backend"        OFF)
option(DRAGONIANLIB_ROCM                    "rocm backend"        OFF)
option(DRAGONIANLIB_SHARED_LIBS             "build shared libs"   OFF)
option(DRAGONIANLIB_ONNXRUNTIME             "build ort libs"      ON)
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
set(CMAKE_CUDA_ARCHITECTURES 89)
# set BLAS_LIBRARIES BLAS_INCLUDE_DIRS FFMPEG_LIBRARIES FFMPEG_INCLUDE_DIRS ONNXRT_LIBRARIES ONNXRT_INCLUDE
# set ONNXRT_DYNLIBS

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
    add_compile_options(/Zi /Gy)
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

file(GLOB_RECURSE DRAGONIANLIB_LIBRARIES_SRC 
    ${PROJECT_SOURCE_DIR}/Libraries/*.cpp
    ${PROJECT_SOURCE_DIR}/Libraries/*.h
    ${PROJECT_SOURCE_DIR}/Libraries/*.c
    ${PROJECT_SOURCE_DIR}/Libraries/*.hpp
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
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.hpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.h
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.cpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.c
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.hpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.h
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.cpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.c
        ${PROJECT_SOURCE_DIR}/Other/*.c
        ${PROJECT_SOURCE_DIR}/Other/*.cpp
        ${PROJECT_SOURCE_DIR}/Other/*.h
        ${PROJECT_SOURCE_DIR}/Other/*.hpp
    )
endif()

set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
#build shared lib?
if(DRAGONIANLIB_SHARED_LIBS)
    add_library(${DRAGONIANLIB_LIB} SHARED ${DRAGONIANLIB_LIBRARIES_SRC} ${DRAGONIANLIB_TENSORLIB_SRC} ${DRAGONIANLIB_ORTLIB_SRC})
    add_definitions(-DDRAGONIANLIB_EXPORT)
    target_compile_definitions(${DRAGONIANLIB_LIB} PRIVATE -DDRAGONIANLIB_EXPORT)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    add_library(${DRAGONIANLIB_LIB} STATIC ${DRAGONIANLIB_LIBRARIES_SRC} ${DRAGONIANLIB_TENSORLIB_SRC} ${DRAGONIANLIB_ORTLIB_SRC})
endif()

#link
set(BUILD_TESTING OFF)
set(BLA_STATIC ON)
set(BLA_VENDOR OpenBLAS)
if(NOT BLAS_LIBRARIES)
    add_subdirectory(Third-Party-Libraries/OpenBLAS)
    target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC openblas)
    set(BLAS_LIBRARIES openblas)
    set(LAPACK_LIBRARIES openblas)
else()
    target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${BLAS_LIBRARIES})
endif()
include_directories(${BLAS_INCLUDE_DIRS})

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

add_subdirectory(Third-Party-Libraries/libmidi)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC libremidi)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/include)

#if(NOT ONNXRT_LIBRARIES)
#    set(ONNXRT_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/onnxruntime/runtimes/win-x64/native/onnxruntime.lib)
#endif()
#if(NOT ONNXRT_INCLUDE)
#    set(ONNXRT_INCLUDE ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/onnxruntime/build/native/include)
#endif()
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core/session)
if(DRAGONIANLIB_ONNXRUNTIME)
    set(onnxruntime_USE_AVX2 ON)
    set(onnxruntime_USE_CUDA ON)
    if(WIN32)
        set(onnxruntime_USE_DML ON)
    endif()
    set(onnxruntime_BUILD_SHARED_LIB ON)
    add_subdirectory(Third-Party-Libraries/ortmain/cmake)
    add_definitions(-DDRAGONIANLIB_ONNXRT_LIB)
    target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC onnxruntime)
    target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC onnxruntime_providers_shared)
else()
    
endif()

#project include
include_directories(${PROJECT_SOURCE_DIR}/include/Base)
if(DRAGONIANLIB_ONNXRUNTIME)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/header)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/header)
endif()
include_directories(${PROJECT_SOURCE_DIR}/Libraries/MyTemplateLibrary)
include_directories(${PROJECT_SOURCE_DIR}/Libraries/AvCodec)
include_directories(${PROJECT_SOURCE_DIR}/Libraries/K-DimensionalTree)
include_directories(${PROJECT_SOURCE_DIR}/Libraries)
include_directories(${PROJECT_SOURCE_DIR}/Other)

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