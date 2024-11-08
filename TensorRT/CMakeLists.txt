cmake_minimum_required (VERSION 3.13)

set(DRAGONIANLIB_TRT_LIB ${DRAGONIANLIB_LIB}TRTLib)

project (
    ${DRAGONIANLIB_TRT_LIB}
    VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
    LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

set(DRAGONIANLIB_TRTLIB_SRC)
file(GLOB_RECURSE DRAGONIANLIB_TRTLIB_SRC
    ${PROJECT_SOURCE_DIR}/*.hpp
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/*.c
)

if(DRAGONIANLIB_SHARED_LIBS)
    add_library(${DRAGONIANLIB_TRT_LIB} SHARED ${DRAGONIANLIB_TRTLIB_SRC})
else()
    add_library(${DRAGONIANLIB_TRT_LIB} STATIC ${DRAGONIANLIB_TRTLIB_SRC})
endif()

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(
		${DRAGONIANLIB_LIB} PRIVATE
		/GL /GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental
	)
	target_link_options(
        ${DRAGONIANLIB_LIB} PRIVATE
        /LTCG
    )
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(
		${DRAGONIANLIB_LIB} PRIVATE
		/GL /Gy  /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /Zi
	)
    target_link_options(
        ${DRAGONIANLIB_LIB} PRIVATE
        /LTCG
    )
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		${DRAGONIANLIB_LIB} PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental
	)
    target_link_options(
        ${DRAGONIANLIB_LIB} PRIVATE
        /LTCG
    )
endif()

if(MSVC)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif()
endif()
