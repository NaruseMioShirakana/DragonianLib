add_executable(SVCNCNN "main.cpp")

if(DRAGONIANLIB_SHARED_LIBS)
	target_compile_definitions(SVCNCNN PRIVATE DRAGONIANLIB_USE_SHARED_LIBS)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		SVCNCNN PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental /bigobj
	)
endif()

target_link_libraries(SVCNCNN PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(SVCNCNN PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

target_link_libraries(SVCNCNN PRIVATE ${DRAGONIANLIB_NCNN_LIBRARIES})
target_include_directories(SVCNCNN PRIVATE ${DRAGONIANLIB_NCNN_INCLUDE_DIRS})

if(DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION)
	target_link_libraries(SVCNCNN PRIVATE ${DRAGONIANLIB_ONNX_LIBRARIES})
	target_include_directories(SVCNCNN PRIVATE ${DRAGONIANLIB_ONNX_INCLUDE_DIRS})
	target_compile_definitions(SVCNCNN PRIVATE DRAGONIANLIB_USE_ORT_CHECK)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET SVCNCNN PROPERTY CXX_STANDARD 23)
endif()

