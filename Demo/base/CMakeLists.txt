add_executable(DragonianLibTest "main.cpp")

if(DRAGONIANLIB_SHARED_LIBS)
	target_compile_definitions(DragonianLibTest PRIVATE DRAGONIANLIB_USE_SHARED_LIBS)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		DragonianLibTest PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental /bigobj
	)
endif()

target_link_libraries(DragonianLibTest PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(DragonianLibTest PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET DragonianLibTest PROPERTY CXX_STANDARD 23)
endif()
