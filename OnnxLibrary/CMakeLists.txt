﻿cmake_minimum_required (VERSION 3.13)

set(DRAGONIANLIB_ORT_LIB ${DRAGONIANLIB_LIB}OrtLib)

project (
	${DRAGONIANLIB_ORT_LIB}
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

set(DRAGONIANLIB_ORTLIB_SRC)
file(GLOB_RECURSE DRAGONIANLIB_ORTLIB_SRC
	${PROJECT_SOURCE_DIR}/*.c
	${PROJECT_SOURCE_DIR}/*.cpp
	${PROJECT_SOURCE_DIR}/*.h
	${PROJECT_SOURCE_DIR}/*.hpp
)

if(DRAGONIANLIB_SHARED_LIBS)
	add_library(${DRAGONIANLIB_ORT_LIB} SHARED ${DRAGONIANLIB_ORTLIB_SRC})
else()
	add_library(${DRAGONIANLIB_ORT_LIB} STATIC ${DRAGONIANLIB_ORTLIB_SRC})
endif()

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(
		${DRAGONIANLIB_ORT_LIB} PRIVATE
		/GL /GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental
	)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(
		${DRAGONIANLIB_ORT_LIB} PRIVATE
		/GL /Gy  /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /Zi
	)
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		${DRAGONIANLIB_ORT_LIB} PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental
	)
endif()

target_include_directories(
	${DRAGONIANLIB_ORT_LIB} PRIVATE
	${DRAGONIANLIB_TENSORLIB_BASE_INCLUDE_DIRS}
	${DRAGONIANLIB_BASE_INCLUDE_DIRS}
	${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(
	${DRAGONIANLIB_ORT_LIB} PRIVATE
	${DRAGONIANLIB_LIB}
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
	set_property(TARGET ${DRAGONIANLIB_ORT_LIB} PROPERTY CXX_STANDARD 23)
endif()

if(MSVC)
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()
endif()

# 导出目标
install(
	TARGETS ${DRAGONIANLIB_ORT_LIB}
	EXPORT ${DRAGONIANLIB_ORT_LIB}Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

# 配置和安装配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_ORT_LIB}ConfigVersion.cmake
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)

configure_file(
	Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_ORT_LIB}Config.cmake
	COPYONLY
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_ORT_LIB}Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_ORT_LIB}ConfigVersion.cmake
	DESTINATION lib/cmake/${DRAGONIANLIB_ORT_LIB}
)
