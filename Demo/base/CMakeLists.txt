add_executable(DragonianLibTest "main.cpp")

if(DRAGONIANLIB_SHARED_LIBS)
	target_compile_definitions(DragonianLibTest PRIVATE DRAGONIANLIB_USE_SHARED_LIBS)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(DragonianLibTest PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(DragonianLibTest PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE_DEB})
else()
	target_compile_options(DragonianLibTest PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_DEBUG})
endif()

target_link_libraries(DragonianLibTest PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(DragonianLibTest PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})
