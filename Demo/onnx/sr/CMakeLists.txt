add_executable(DragonianLibSR "main.cpp")

if(DRAGONIANLIB_SHARED_LIBS)
	target_compile_definitions(DragonianLibSR PRIVATE DRAGONIANLIB_USE_SHARED_LIBS)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		DragonianLibSR PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental /bigobj
	)
endif()

target_link_libraries(DragonianLibSR PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(DragonianLibSR PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

target_link_libraries(DragonianLibSR PRIVATE ${DRAGONIANLIB_ONNX_LIBRARIES})
target_include_directories(DragonianLibSR PRIVATE ${DRAGONIANLIB_ONNX_INCLUDE_DIRS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET DragonianLibSR PROPERTY CXX_STANDARD 23)
endif()

