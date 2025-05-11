include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DRAGONIANLIBDEMOEXE DragonianLibDemo)

if(DRAGONIANLIB_SHARED_LIBS)
	message("Build Shared Lib Demo")
	add_definitions(-DDRAGONIANLIB_IMPORT)
endif()

if (false)
	file(GLOB_RECURSE F0EditorUISrc 
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.c
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/UI/*.cpp
	)
	add_executable(${DRAGONIANLIBDEMOEXE} ${F0EditorUISrc})
	include_directories("D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/src/include")
	if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
		target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE "D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/library/MiaoUI64d.lib")
	else()
		target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE "D:/VSGIT/MoeVS-SVC/Lib/MiaoUI/MiaoUI/library/MiaoUI64.lib")
	endif()
else()
	add_executable(${DRAGONIANLIBDEMOEXE} "SVC-TensorRT.cpp")
endif()

#Build Type
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		/GL /GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /bigobj
	)
	target_link_options(
        ${DRAGONIANLIBDEMOEXE} PRIVATE
        /LTCG
    )
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		
	)
	target_link_options(
        ${DRAGONIANLIBDEMOEXE} PRIVATE
        /LTCG
    )
else()
	add_definitions(-DDRAGONIANLIB_DEBUG)
	target_compile_options(
		${DRAGONIANLIBDEMOEXE} PRIVATE
		/Zi /Gy /experimental:module /openmp:experimental /bigobj
	)
endif()

target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS})

#target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_ONNX_LIBRARIES})
#target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_ONNX_INCLUDE_DIRS}) 

target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_TENSORRT_LIBRARIES})
target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_TENSORRT_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS}) 


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${DRAGONIANLIBDEMOEXE} PROPERTY CXX_STANDARD 23)
endif()
