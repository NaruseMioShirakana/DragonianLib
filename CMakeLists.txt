cmake_minimum_required (VERSION 3.13)

#Options
option(LIVSVC_DML                     "dml backend"         OFF)
option(LIVSVC_CUDA                    "cuda backend"        ON)
option(LIVSVC_ROCM                    "rocm backend"        OFF)
option(LIVSVC_SHARED_LIBS             "build shared libs"   OFF)
option(LIVSVC_ONNXRUNTIME             "build ort libs"      OFF)
option(LIBSVC_BUILD_DEMO              "build demo"          ON)
add_definitions(-DLIBSVC_ALLOC_ALIG=32)
add_definitions(-DLIBSVC_ALIG_DIM_SHAPE=8)  #Don't Change This Value'
add_definitions(-DLIBSVC_CONT_THRESHOLD_FRONT=8)
add_definitions(-DLIBSVC_CONT_THRESHOLD_BACK=32)
add_definitions(-DLIBSVC_EMPTY_CAPACITY=16)
add_definitions(-DLIBSVC_CONT_THRESHOLD_MIN_SIZE=65536)
add_definitions(-DLIBSVC_NAME_MAX_SIZE=1024)
add_definitions(-DUNICODE)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_ENABLE_GPU OFF)
set(LIBSVC_LANGUAGES CXX)

#functions
function(LibSvcAddFiles _out)
    file(GLOB_RECURSE _tmp ${ARGN})
    list(APPEND ${_out} ${_tmp})
    set(${_out} ${${_out}} PARENT_SCOPE)
endfunction()

#project name
set(LIBSVC_LIB libsvc)

#device
if(LIVSVC_ONNXRUNTIME)
    message("Onnx Build")
else()
    if(LIVSVC_CUDA)
        add_definitions(-DLIBSVC_ENABLECUDA)
        set(FAISS_ENABLE_GPU ON)
        list(APPEND LIBSVC_LANGUAGES CUDA)
    endif()
    if (LIVSVC_ROCM)
        add_definitions(-DLIBSVC_ENABLEROCM)
        list(APPEND LIBSVC_LANGUAGES HIP)
    endif ()
    if (LIVSVC_DML)
        add_definitions(-DLIBSVC_ENABLEDML)
    endif ()
endif ()

#project
project (
    ${LIBSVC_LIB} 
    VERSION 1.0.0
    LANGUAGES ${LIBSVC_LANGUAGES}
)

if(NOT DEFINED FFMPEG_LIBRARIES)
    set(FFMPEG_LIBRARIES)
    set(LIBSVC_FFMPEG_OUTDIR "Third-Party-Libraries/ffmpeg/bin")
    list(
        APPEND FFMPEG_LIBRARIES
        ${PROJECT_SOURCE_DIR}/${LIBSVC_FFMPEG_OUTDIR}/avcodec.lib
        ${PROJECT_SOURCE_DIR}/${LIBSVC_FFMPEG_OUTDIR}/avformat.lib
        ${PROJECT_SOURCE_DIR}/${LIBSVC_FFMPEG_OUTDIR}/avutil.lib
        ${PROJECT_SOURCE_DIR}/${LIBSVC_FFMPEG_OUTDIR}/swresample.lib
        ${PROJECT_SOURCE_DIR}/${LIBSVC_FFMPEG_OUTDIR}/swscale.lib
    )
endif()
if(NOT DEFINED FFMPEG_INCLUDE_DIRS)
    set(FFMPEG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg)
endif()

if(NOT DEFINED BLAS_LIBRARIES)
    set(BLAS_LIBRARIES)
endif()
if(NOT DEFINED BLAS_INCLUDE_DIRS)
    set(BLAS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS)
endif()

#outdirs
if(LIVSVC_ONNXRUNTIME)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/ort/archive/${CMAKE_BUILD_TYPE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/ort/library/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/ort/runtime/${CMAKE_BUILD_TYPE})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/out/ort/lib/${CMAKE_BUILD_TYPE})
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/archive/${CMAKE_BUILD_TYPE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/library/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/runtime/${CMAKE_BUILD_TYPE})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/out/lib/${CMAKE_BUILD_TYPE})
endif ()

#bulid type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message("Build Release")
    add_compile_options(/GS- /arch:AVX2 /O2 /Ob2 /Ot)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    add_compile_options(/Zi /Gy)
else()
    message("Build Debug")
    add_definitions(-DLIBSVC_DEBUG)
    add_compile_options(/Zi /Gy)
endif()

if(MSVC)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif()
endif()

#src
set(PROJECT_SOURCES)
if(LIVSVC_ONNXRUNTIME)
    LibSvcAddFiles(PROJECT_SOURCES
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.hpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.h
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.cpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/*.c
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.hpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.h
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.cpp
        ${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/*.c
        ${PROJECT_SOURCE_DIR}/Libraries/*.cpp
        ${PROJECT_SOURCE_DIR}/Libraries/*.h
        ${PROJECT_SOURCE_DIR}/Libraries/*.c
        ${PROJECT_SOURCE_DIR}/Libraries/*.hpp
    )
else()
    LibSvcAddFiles(PROJECT_SOURCES
        ${PROJECT_SOURCE_DIR}/include/*.hpp
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.c
        ${PROJECT_SOURCE_DIR}/Libraries/*.cpp
        ${PROJECT_SOURCE_DIR}/Libraries/*.h
        ${PROJECT_SOURCE_DIR}/Libraries/*.c
        ${PROJECT_SOURCE_DIR}/Libraries/*.hpp
    )
endif()

set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
#build shared lib?
if(LIVSVC_SHARED_LIBS)
    add_library(${LIBSVC_LIB} SHARED ${PROJECT_SOURCES})
    add_definitions(-DLIBSVC_EXPORT)
    target_compile_definitions(${LIBSVC_LIB} PRIVATE -DLIBSVC_EXPORT)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    add_library(${LIBSVC_LIB} STATIC ${PROJECT_SOURCES})
endif()

#link
set(BUILD_TESTING OFF)
set(BLA_STATIC ON)
set(BLA_VENDOR OpenBLAS)
if(NOT BLAS_LIBRARIES)
    add_subdirectory(Third-Party-Libraries/OpenBLAS)
    target_link_libraries(${LIBSVC_LIB} PUBLIC openblas)
else()
    target_link_libraries(${LIBSVC_LIB} PUBLIC ${BLAS_LIBRARIES})
endif()
include_directories(${BLAS_INCLUDE_DIRS})

target_link_libraries(${LIBSVC_LIB} PUBLIC ${FFMPEG_LIBRARIES})
include_directories(${FFMPEG_INCLUDE_DIRS})

add_subdirectory(Third-Party-Libraries/World)
target_link_libraries(${LIBSVC_LIB} PUBLIC world)
target_link_libraries(${LIBSVC_LIB} PUBLIC world_tool)
include_directories(${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/src/world)



if(LIVSVC_ONNXRUNTIME)
    
else()
    
endif()

#project include
if(LIVSVC_ONNXRUNTIME)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx/Modules/header)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx)
    include_directories(${PROJECT_SOURCE_DIR}/libsvc-onnx/Api/header)
else()
    include_directories(${PROJECT_SOURCE_DIR}/include/Base)
endif()
include_directories(${PROJECT_SOURCE_DIR}/Libraries/MyTemplateLibrary)
include_directories(${PROJECT_SOURCE_DIR}/Libraries/AvCodec)
include_directories(${PROJECT_SOURCE_DIR}/Libraries/K-DimensionalTree)

#standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET libsvc PROPERTY CXX_STANDARD 23)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
    target_link_options(${LIBSVC_LIB} PRIVATE 
        /NATVIS:${PROJECT_SOURCE_DIR}/NatvisFile.natvis
    )
endif()

#demo
if (LIBSVC_BUILD_DEMO)
    if(LIVSVC_ONNXRUNTIME)
        set(LibSvcOrtDemo on)
    endif()
    add_subdirectory(Demo)
endif()