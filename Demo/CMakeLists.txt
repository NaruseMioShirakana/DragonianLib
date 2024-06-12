include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(LibSvcOrtDemo "" off)

set(EXE_DEMO Demo)
add_compile_options(/ZI)

if(LibSvcOrtDemo)
	message("Build Ort Demo")
	add_executable(${EXE_DEMO} "mainort.cpp")
else()
	message("Build Lib Demo")
	add_executable(${EXE_DEMO} "main.cpp")
endif()

target_link_libraries(${EXE_DEMO} PRIVATE DragonianLib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${EXE_DEMO} PROPERTY CXX_STANDARD 20)
endif()