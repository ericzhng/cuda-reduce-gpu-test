
##===============================================================================
# Set source file and compile
##===============================================================================
SET(SRC
	cpu_subs.cpp
	cpu_subs.h
	)

# create the library
add_library(cpu_code STATIC ${SRC})

# must use separable build
set_target_properties(cpu_code PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	# COMPILE_FLAGS ${CMAKE_CXX_CUDA_FLAGS}
	LINKER_LANGUAGE CXX
	)
