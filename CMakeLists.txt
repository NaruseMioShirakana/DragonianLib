cmake_minimum_required (VERSION 3.13)

#Global Visual Studio Macros
add_definitions(-DUNICODE)
add_definitions(-D_UNICODE)
add_definitions(-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)

#DragonianLib Build Options
option(DRAGONIANLIB_DML                     "dml backend"         OFF)
option(DRAGONIANLIB_CUDA                    "cuda backend"        OFF)
option(DRAGONIANLIB_ROCM                    "rocm backend"        OFF)
option(DRAGONIANLIB_SHARED_LIBS             "build shared libs"   OFF)
option(DRAGONIANLIB_ONNXRUNTIME             "build ort libs"      ON)
option(DRAGONIANLIB_TENSORRT                "build trt libs"      OFF)
option(DRAGONIANLIB_STATIC_FFMPEG           "use static ffmpeg"   ON)
option(DRAGONIANLIB_BUILD_DEMO              "build demo"          ON)
if(DRAGONIANLIB_TENSORRT)
	set(DRAGONIANLIB_CUDA ON)
endif()

#DragonianLib Configs
add_definitions(-DDRAGONIANLIB_ALLOC_ALIG=32)
add_definitions(-DDRAGONIANLIB_ALIG_DIM_SHAPE=8)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_FRONT=8)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_BACK=32)
add_definitions(-DDRAGONIANLIB_EMPTY_CAPACITY=16)
add_definitions(-DDRAGONIANLIB_PADDING_COUNT=64000)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE=65536)
add_definitions(-DDRAGONIANLIB_NAME_MAX_SIZE=1024)

#Global Config
set(BUILD_TESTING OFF)
set(BUILD_BENCHMARKS OFF)

#Project Name
set(DRAGONIANLIB_LIB DragonianLib)

#Languages
set(DRAGONIANLIB_LANGUAGES CXX ASM_MASM)

#Execusion Providers
if(DRAGONIANLIB_CUDA)
	add_definitions(-DDRAGONIANLIB_ENABLECUDA)
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

#Version
set(DRAGONIANLIB_VERSION_MAJOR 1)
set(DRAGONIANLIB_VERSION_MINOR 0)
set(DRAGONIANLIB_VERSION_PATCH 0)

#Project
project (
	${DRAGONIANLIB_LIB}
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

#Set FFMPEG Libraries And Include Directories
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

#Find OpenBLAS Libraries And Include Directories
find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
	if(NOT DEFINED OpenBLAS_LIBRARIES)
		set(OpenBLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/lib/openblas.lib)
	endif()
	if(NOT DEFINED OpenBLAS_INCLUDE_DIRS)
		set(OpenBLAS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/include/openblas)
	endif()
endif()

#Set CUDA Libraries And Include Directories
if(DRAGONIANLIB_CUDA)
	find_package(libcudacxx REQUIRED)
	if(libcudacxx_FOUND)
		set(CUDA_LIBRARIES
			libcudacxx::libcudacxx
			${_libcudacxx_INCLUDE_DIR}/../lib/x64/cudart_static.lib
		)
	elseif(NOT DEFINED CUDA_LIBRARIES OR NOT DEFINED CUDA_INCLUDE_DIRS)
		message(FATAL_ERROR "CUDA Libraries Not Found")
	endif()
endif()

#Find LibreMidi Libraries And Include Directories
find_package(libremidi)
if(NOT libremidi_FOUND)
	if(NOT DEFINED LIBREMIDI_LIBRARIES)
		set(LIBREMIDI_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/out/install/x64-Release/lib/libremidi.lib)
	endif()
	if(NOT DEFINED LIBREMIDI_INCLUDE_DIRS)
		set(LIBREMIDI_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/out/install/x64-Release/include)
	endif()
endif()

#Find FFTW Libraries And Include Directories
find_package(FFTW3)
if(NOT FFTW3_FOUND)
	if(NOT DEFINED FFTW3_LIBRARIES)
		set(FFTW3_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/fftw-3.3.10/out/install/x64-Release/lib/fftw3.lib)
	endif()
	if(NOT DEFINED FFTW3_INCLUDE_DIRS)
		set(FFTW3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/fftw-3.3.10/out/install/x64-Release/include)
	endif()
endif()

#Find World Libraries And Include Directories
find_package(world)
if(NOT world_FOUND)
	if(NOT DEFINED WORLD_LIBRARIES)
		set(WORLD_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/out/install/x64-Release/lib/world.lib)
	endif()
	if(NOT DEFINED WORLD_INCLUDE_DIRS)
		set(WORLD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/out/install/x64-Release/include)
	endif()
endif()

#Find Faiss Libraries And Include Directories
find_package(faiss)
if(NOT faiss_FOUND)
	if(NOT DEFINED FAISS_LIBRARIES)
		set(FAISS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Release/lib/faiss_avx2.lib)
	endif()
	if(NOT DEFINED FAISS_INCLUDE_DIRS)
		set(FAISS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Release/include)
	endif()
endif()

#Set OnnxRuntime Libraries And Include Directories
if(DRAGONIANLIB_ONNXRUNTIME)
	if(NOT DEFINED ONNXRUNTIME_LIBRARIES)
		set(
			ONNXRUNTIME_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_shared.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_dml.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime_providers_cuda.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/Release/Release/onnxruntime.lib
		)
	endif()
	if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIRS)
		set(
			ONNXRUNTIME_INCLUDE_DIRS
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core/session
		)
	endif()
endif()

#Set TensorRT Libraries And Include Directories
if(DRAGONIANLIB_TENSORRT)
	if(NOT DEFINED TENSORRT_LIBRARIES)
		set(
			TENSORRT_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_dispatch_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_lean_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_plugin_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_vc_plugin_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvonnxparser_10.lib
		)
	endif()
	if(NOT DEFINED TENSORRT_INCLUDE_DIRS)
		set(TENSORRT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/include)
	endif()
endif()

#Output Directories
if(DRAGONIANLIB_SHARED_LIBS)
	set(DRAGONIANLIB_TYPE_DIR "Shared")
else()
	set(DRAGONIANLIB_TYPE_DIR "Static")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Runtime)
set(LIBRARY_OUTPUT_PATH             ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/OLibrary)

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	message("Build Release")
	add_compile_options(/GS- /arch:AVX2 /O2 /Ot)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	add_compile_options(/Zi /Gy /arch:AVX2 /O2 /Ot)
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
file(GLOB_RECURSE DRAGONIANLIB_LIBRARIES_SRC 
	${PROJECT_SOURCE_DIR}/Libraries/*.cpp
	${PROJECT_SOURCE_DIR}/Libraries/*.h
	${PROJECT_SOURCE_DIR}/Libraries/*.c
	${PROJECT_SOURCE_DIR}/Libraries/*.hpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.hpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.h
	${PROJECT_SOURCE_DIR}/TensorLib/*.cpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.c
	#${PROJECT_SOURCE_DIR}/Libraries/*.asm
)

#Source code
if(DRAGONIANLIB_SHARED_LIBS)
	add_library(${DRAGONIANLIB_LIB} SHARED ${DRAGONIANLIB_LIBRARIES_SRC})
	add_definitions(-DDRAGONIANLIB_EXPORT)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
	add_library(${DRAGONIANLIB_LIB} STATIC ${DRAGONIANLIB_LIBRARIES_SRC})
endif()

#Link Libraries
if(DEFINED FFMPEG_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FFMPEG_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FFMPEG_INCLUDE_DIRS})
else()
	message(FATAL_ERROR "FFMPEG Libraries Not Found")
endif()

if(DEFINED LIBREMIDI_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${LIBREMIDI_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${LIBREMIDI_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC libremidi)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE $<TARGET_PROPERTY:libremidi,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(DEFINED FFTW3_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FFTW3_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FFTW3_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC fftw3)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE $<TARGET_PROPERTY:fftw3,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(DEFINED WORLD_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${WORLD_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${WORLD_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC world)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE $<TARGET_PROPERTY:world,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(DEFINED FAISS_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FAISS_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FAISS_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC faiss_avx2)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE $<TARGET_PROPERTY:faiss_avx2,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(DEFINED OpenBLAS_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${OpenBLAS_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC OpenBLAS::OpenBLAS)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE $<TARGET_PROPERTY:OpenBLAS::OpenBLAS,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(DRAGONIANLIB_CUDA)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${_libcudacxx_INCLUDE_DIR})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${CUDA_LIBRARIES})
endif()

add_subdirectory(Third-Party-Libraries/yyjson)
target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC yyjson)
target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/yyjson/src)

#project include
set(DRAGONIANLIB_TENSORLIB_BASE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/TensorLib/Include/Base)
set(DRAGONIANLIB_BASE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Libraries)
target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${PROJECT_SOURCE_DIR}/TensorLib/Include/Base)
target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${PROJECT_SOURCE_DIR}/Libraries)

if(DRAGONIANLIB_ONNXRUNTIME)
	add_definitions(-DDRAGONIANLIB_ONNXRT_LIB)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${ONNXRUNTIME_LIBRARIES})
	add_subdirectory(OnnxLibrary)
endif()

if(DRAGONIANLIB_TENSORRT)
	add_definitions(-DDRAGONIANLIB_TENSORRT_LIB)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${TENSORRT_INCLUDE_DIRS})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${TENSORRT_LIBRARIES})
endif()

#C++ Standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET ${DRAGONIANLIB_LIB} PROPERTY CXX_STANDARD 23)
endif()

#Natvis File
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
	target_link_options(${DRAGONIANLIB_LIB} PRIVATE 
		/NATVIS:${PROJECT_SOURCE_DIR}/NatvisFile.natvis
	)
endif()

set(DRAGONIANLIB_LIBRARIES
	${DRAGONIANLIB_LIB}
	${DRAGONIANLIB_ORT_LIB}
	${DRAGONIANLIB_TRT_LIB})

set(DRAGONIANLIB_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/Libraries
	${PROJECT_SOURCE_DIR}/OnnxLibrary
	${PROJECT_SOURCE_DIR}/TensorRT
	${PROJECT_SOURCE_DIR}/TensorLib)

# 安装目标
install(
	TARGETS ${DRAGONIANLIB_LIB}
	EXPORT ${DRAGONIANLIB_LIB}Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

# 安装头文件
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/Libraries/
	DESTINATION include/Libraries
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/OnnxLibrary/
	DESTINATION include/OnnxLibrary
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/TensorRT/
	DESTINATION include/TensorRT
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/TensorLib/
	DESTINATION include/TensorLib
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)

# 导出子目录中的目标
install(
	EXPORT ${DRAGONIANLIB_LIB}Targets
	FILE ${DRAGONIANLIB_LIB}Targets.cmake
	NAMESPACE ${DRAGONIANLIB_LIB}::
	DESTINATION lib/cmake/${DRAGONIANLIB_LIB}
)

if(DRAGONIANLIB_ONNXRUNTIME)
	set(DRAGONIANLIB_ORT_LIB ${DRAGONIANLIB_LIB}OrtLib)
	install(
		EXPORT ${DRAGONIANLIB_ORT_LIB}Targets
		FILE ${DRAGONIANLIB_ORT_LIB}Targets.cmake
		NAMESPACE ${DRAGONIANLIB_LIB}::
		DESTINATION lib/cmake/${DRAGONIANLIB_ORT_LIB}
	)
endif()

if(DRAGONIANLIB_TENSORRT)
	set(DRAGONIANLIB_TRT_LIB ${DRAGONIANLIB_LIB}TRTLib)
	install(
		EXPORT ${DRAGONIANLIB_TRT_LIB}Targets
		FILE ${DRAGONIANLIB_TRT_LIB}Targets.cmake
		NAMESPACE ${DRAGONIANLIB_LIB}::
		DESTINATION lib/cmake/${DRAGONIANLIB_TRT_LIB}
	)
endif()

# 配置和安装配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}ConfigVersion.cmake
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}Config.cmake
	COPYONLY
)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}ConfigVersion.cmake
	DESTINATION lib/cmake/${DRAGONIANLIB_LIB}
)

if(DRAGONIANLIB_ONNXRUNTIME)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/OnnxLibrary/${DRAGONIANLIB_ORT_LIB}Config.cmake
		${CMAKE_CURRENT_BINARY_DIR}/OnnxLibrary/${DRAGONIANLIB_ORT_LIB}ConfigVersion.cmake
		DESTINATION lib/cmake/${DRAGONIANLIB_ORT_LIB}
	)
endif()

if(DRAGONIANLIB_TENSORRT)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/TensorRT/${DRAGONIANLIB_TRT_LIB}Config.cmake
		${CMAKE_CURRENT_BINARY_DIR}/TensorRT/${DRAGONIANLIB_TRT_LIB}ConfigVersion.cmake
		DESTINATION lib/cmake/${DRAGONIANLIB_TRT_LIB}
	)
endif()

#Build Demo
if (DRAGONIANLIB_BUILD_DEMO)
	if(DRAGONIANLIB_ONNXRUNTIME)
		set(DRAGONIANLIBORTDEMO ON)
	endif()
	add_subdirectory(Demo)
endif()
