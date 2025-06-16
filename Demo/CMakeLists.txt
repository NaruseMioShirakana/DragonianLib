

if(DRAGONIANLIB_SHARED_LIBS)
	message("Build Shared Lib Demo")
	add_definitions(-DDRAGONIANLIB_IMPORT)
endif()

if (false)
	set(DRAGONIANLIBDEMOEXE DragonianLibDemo)
	file(GLOB_RECURSE F0EditorUISrc 
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.c
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.cpp
	)
	add_executable(${DRAGONIANLIBDEMOEXE} ${F0EditorUISrc})
	target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE "D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/src/include")
	if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
		target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE "D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/library/MiaoUI64d.lib")
	else()
		target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE "D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/library/MiaoUI64.lib")
	endif()

	#Build Type
	if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	else()
		add_definitions(-DDRAGONIANLIB_DEBUG)
		target_compile_options(
			${DRAGONIANLIBDEMOEXE} PRIVATE
			/Zi /Gy /experimental:module /openmp:experimental /bigobj
		)
	endif()

	target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_LIBRARIES})
	target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

	target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_ONNX_LIBRARIES})
	target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_ONNX_INCLUDE_DIRS}) 
endif()

if (DRAGONIANLIB_ORT_SUPER_RESOLUTION)
	message("Build Onnx Super Resolution Demo")
	add_subdirectory(onnx/sr)
endif()
if (DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION)
	message("Build Onnx Singing Voice Conversion Demo")
	add_subdirectory(onnx/svc)
endif()
if (DRAGONIANLIB_ORT_TEXT_TO_SPEECH)
	message("Build Onnx Text to Speech Demo")
	add_subdirectory(onnx/tts)
endif()
if (DRAGONIANLIB_ORT_MUSIC_TRANSCRIPTION)
	message("Build Onnx Music Transcription Demo")
	add_subdirectory(onnx/mt)
endif()
if (DRAGONIANLIB_ORT_DEMIX)
	message("Build Onnx Demix Demo")
	add_subdirectory(onnx/demix)
endif()
if (DRAGONIANLIB_NCNN_SINGING_VOICE_CONVERSION)
	message("Build NCNN Singing Voice Conversion Demo")
	add_subdirectory(ncnn/svc)
endif()
add_subdirectory(base)