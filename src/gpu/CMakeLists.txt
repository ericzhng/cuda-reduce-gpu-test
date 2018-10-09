
##===============================================================================
# Set source file and compile
##===============================================================================

set(MAIN_SRC
	subroutines.cu
	subroutines.cuh
	main.cu
	)

# create the executable
add_executable(${TARGET_NAME} ${MAIN_SRC})

add_definitions(-DCRT_SECURE_NO_DEPRECATE)

# must use separable build
set_target_properties(${TARGET_NAME} PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	COMPILE_FLAGS ${CMAKE_CXX_CUDA_FLAGS}
	)
	
# get rid of debug and release folders
#if(WIN32)
	#set_target_properties(${TARGET_NAME} PROPERTIES PREFIX “../” )
	#set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "_d")
#endif()

# link libraries
target_link_libraries(${TARGET_NAME}
	cpu_code
	)
