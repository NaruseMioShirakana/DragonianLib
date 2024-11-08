include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DRAGONIANLIBDEMOEXE DragonianLibDemo)

if(DRAGONIANLIB_SHARED_LIBS)
	message("Build Shared Lib Demo")
	add_definitions(-DDRAGONIANLIB_IMPORT)
endif()

if(DRAGONIANLIBORTDEMO)
	#message("Build Ort Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "mainort.cpp")
else()
	#message("Build Lib Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "main.cpp")
endif()

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		/GL /GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental
	)
	target_link_options(
        ${DRAGONIANLIBDEMOEXE} PRIVATE
        /LTCG
    )
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		/GL /Gy  /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /Zi
	)
	target_link_options(
        ${DRAGONIANLIBDEMOEXE} PRIVATE
        /LTCG
    )
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental
	)
	target_link_options(
        ${DRAGONIANLIBDEMOEXE} PRIVATE
        /LTCG
    )
endif()

message(${DRAGONIANLIB_LIBRARIES})

target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS}) 

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${DRAGONIANLIBDEMOEXE} PROPERTY CXX_STANDARD 23)
endif()
