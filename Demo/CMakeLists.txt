include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DRAGONIANLIBDEMOEXE DragonianLibDemo)
add_compile_options(/ZI)

if(DRAGONIANLIB_SHARED_LIBS)
	message("Build Shared Lib Demo")
	add_definitions(-DDRAGONIANLIB_IMPORT)
endif()

if(DRAGONIANLIBORTDEMO)
	message("Build Ort Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "mainort.cpp")
else()
	message("Build Lib Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "main.cpp")
endif()

message(${DRAGONIANLIB_LIBRARIES})

target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_LIBRARIES})
target_include_directories(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_INCLUDE_DIRS}) 

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${DRAGONIANLIBDEMOEXE} PROPERTY CXX_STANDARD 23)
endif()
