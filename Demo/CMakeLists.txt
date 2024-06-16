include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(DRAGONIANLIBORTDEMO	"" OFF)

set(DRAGONIANLIBDEMOEXE DragonianLibDemo)
add_compile_options(/ZI)

if(DRAGONIANLIB_SHARED_LIBS)
	add_definitions(-DDRAGONIANLIB_IMPORT)
endif()

if(DRAGONIANLIBORTDEMO)
	message("Build Ort Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "mainort.cpp")
else()
	message("Build Lib Demo")
	add_executable(${DRAGONIANLIBDEMOEXE} "main.cpp")
endif()

target_link_libraries(${DRAGONIANLIBDEMOEXE} PRIVATE ${DRAGONIANLIB_LIB})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${DRAGONIANLIBDEMOEXE} PROPERTY CXX_STANDARD 23)
endif()