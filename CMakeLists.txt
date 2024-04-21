cmake_minimum_required (VERSION 3.8)


option(LIVSVC_CUDA                    "cuda backend" OFF)
option(LIVSVC_ROCM                    "rocm backend" OFF)
option(LIVSVC_SHARED_LIBS             "build shared libs" OFF)
option(SVC_BUILD_DEMO                 "SVC_BUILD_DEMO" ON)
add_definitions(-DLIBSVC_ALLOC_ALIG=32)
add_definitions(-DLIBSVC_CONT_THRESHOLD_FRONT=8)
add_definitions(-DLIBSVC_CONT_THRESHOLD_BACK=262144)
add_definitions(-DUNICODE)
add_definitions(-DLIBSVC_DEBUG)

set(LIBSVC_LIB libsvc)
set(BUILD_SHARED_LIBS OFF)
if(LIVSVC_CUDA)
    
endif()

if (LIVSVC_ROCM)
    
endif ()

project (${LIBSVC_LIB})


if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
message("Build Release")
add_compile_options(/GS- /arch:AVX2 /O2 /Ob2)
else()
message("Build Debug")
add_compile_options(/ZI /Gy)
endif()

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

function(LibSvcAddFiles _out)
    file(GLOB_RECURSE _tmp ${ARGN})
    list(APPEND ${_out} ${_tmp})
    set(${_out} ${${_out}} PARENT_SCOPE)
endfunction()

set(PROJECT_SOURCES)
LibSvcAddFiles(PROJECT_SOURCES
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.c
)
if(LIVSVC_SHARED_LIBS)
    add_library(${LIBSVC_LIB} SHARED ${PROJECT_SOURCES})
    add_definitions(-DLIBSVC_EXPORT)
    target_compile_definitions(${LIBSVC_LIB} PRIVATE -DLIBSVC_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    add_library(${LIBSVC_LIB} STATIC ${PROJECT_SOURCES})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/Base)
include_directories(${PROJECT_SOURCE_DIR}/ggml/include)

add_subdirectory(ggml)

target_link_libraries(${LIBSVC_LIB} PUBLIC ggml)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET libsvc PROPERTY CXX_STANDARD 20)
endif()

if (SVC_BUILD_DEMO)
    add_subdirectory(Demo)
endif()