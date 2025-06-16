add_executable(DragonianLibSVC "main.cpp")

if(DRAGONIANLIB_SHARED_LIBS)
	target_compile_definitions(DragonianLibSVC PRIVATE DRAGONIANLIB_USE_SHARED_LIBS)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(DragonianLibSVC PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(DragonianLibSVC PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE_DEB})
else()
	target_compile_options(DragonianLibSVC PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_DEBUG})
endif()

target_link_libraries(DragonianLibSVC PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(DragonianLibSVC PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

target_link_libraries(DragonianLibSVC PRIVATE ${DRAGONIANLIB_ONNX_LIBRARIES})
target_include_directories(DragonianLibSVC PRIVATE ${DRAGONIANLIB_ONNX_INCLUDE_DIRS})
