cmake_minimum_required (VERSION 3.13)

set(DRAGONIANLIB_TRT_LIB ${DRAGONIANLIB_LIB}TRTLib)

project (
    ${DRAGONIANLIB_TRT_LIB}
    VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
    LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_SRC
	${PROJECT_SOURCE_DIR}/TensorRTBase/*.c
	${PROJECT_SOURCE_DIR}/TensorRTBase/*.cpp
	${PROJECT_SOURCE_DIR}/TensorRTBase/*.h
	${PROJECT_SOURCE_DIR}/TensorRTBase/*.hpp
)

if(DRAGONIANLIB_TRT_SINGING_VOICE_CONVERSION)
	file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_SVC_SRC
		${PROJECT_SOURCE_DIR}/SingingVoiceConversion/*.c
		${PROJECT_SOURCE_DIR}/SingingVoiceConversion/*.cpp
		${PROJECT_SOURCE_DIR}/SingingVoiceConversion/*.h
		${PROJECT_SOURCE_DIR}/SingingVoiceConversion/*.hpp
	)
	list(APPEND DRAGONIANLIB_TRTLIB_SRC ${DRAGONIANLIB_TRTLIB_SVC_SRC})
	message("	- Build Singing Voice Conversion")
endif()
if(DRAGONIANLIB_TRT_TEXT_TO_SPEECH)
	file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_TTS_SRC
		${PROJECT_SOURCE_DIR}/TextToSpeech/*.c
		${PROJECT_SOURCE_DIR}/TextToSpeech/*.cpp
		${PROJECT_SOURCE_DIR}/TextToSpeech/*.h
		${PROJECT_SOURCE_DIR}/TextToSpeech/*.hpp
	)
	list(APPEND DRAGONIANLIB_TRTLIB_SRC ${DRAGONIANLIB_TRTLIB_TTS_SRC})
	message("	- Build Text To Speech")
endif()
if(DRAGONIANLIB_TRT_SUPER_RESOLUTION)
 	file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_SR_SRC
		${PROJECT_SOURCE_DIR}/SuperResolution/*.c
		${PROJECT_SOURCE_DIR}/SuperResolution/*.cpp
		${PROJECT_SOURCE_DIR}/SuperResolution/*.h
		${PROJECT_SOURCE_DIR}/SuperResolution/*.hpp
	)
	list(APPEND DRAGONIANLIB_TRTLIB_SRC ${DRAGONIANLIB_TRTLIB_SR_SRC})
	message("	- Build Super Resolution")
endif()
if(DRAGONIANLIB_TRT_MUSIC_TRANSCRIPTION)
 	file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_MT_SRC
		${PROJECT_SOURCE_DIR}/MusicTranscription/*.c
		${PROJECT_SOURCE_DIR}/MusicTranscription/*.cpp
		${PROJECT_SOURCE_DIR}/MusicTranscription/*.h
		${PROJECT_SOURCE_DIR}/MusicTranscription/*.hpp
	)
	list(APPEND DRAGONIANLIB_TRTLIB_SRC ${DRAGONIANLIB_TRTLIB_MT_SRC})
	message("	- Build Music Transcription")
endif()

if(DRAGONIANLIB_SHARED_LIBS)
    add_library(${DRAGONIANLIB_TRT_LIB} SHARED ${DRAGONIANLIB_TRTLIB_SRC})
else()
    add_library(${DRAGONIANLIB_TRT_LIB} STATIC ${DRAGONIANLIB_TRTLIB_SRC})
endif()

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(
		${DRAGONIANLIB_TRT_LIB} PRIVATE
		/GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental
	)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(
		${DRAGONIANLIB_TRT_LIB} PRIVATE
		/Gy  /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /Zi
	)
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		${DRAGONIANLIB_TRT_LIB} PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental
	)
endif()

target_include_directories(
	${DRAGONIANLIB_TRT_LIB} PRIVATE
	${DRAGONIANLIB_TENSORLIB_BASE_INCLUDE_DIRS}
	${DRAGONIANLIB_BASE_INCLUDE_DIRS}
	${TENSORRT_INCLUDE_DIRS}
)

target_link_libraries(
	${DRAGONIANLIB_TRT_LIB} PRIVATE
	${DRAGONIANLIB_LIB}
	${TENSORRT_LIBRARIES}
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET ${DRAGONIANLIB_TRT_LIB} PROPERTY CXX_STANDARD 23)
endif()

if(MSVC)
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()
endif()

# 导出目标
install(
	TARGETS ${DRAGONIANLIB_TRT_LIB}
	EXPORT ${DRAGONIANLIB_TRT_LIB}Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

# 配置和安装配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_TRT_LIB}ConfigVersion.cmake
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)

configure_file(
	Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_TRT_LIB}Config.cmake
	COPYONLY
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_TRT_LIB}Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_TRT_LIB}ConfigVersion.cmake
	DESTINATION lib/cmake/${DRAGONIANLIB_TRT_LIB}
)