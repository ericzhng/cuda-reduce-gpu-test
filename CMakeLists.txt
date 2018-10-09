#===============================================================================
#                                            
# Local Interaction Simulation Approach
#
# CMake Build File
#
# Author: HUI ZHANG
# Date:   Aug. 7, 2017
#===============================================================================

SET(TARGET gpu_reduce)
project(${TARGET} LANGUAGES CXX CUDA)
# project(${TARGET} LANGUAGES CXX)

#===============================================================================
# Check for debug or release mode
#===============================================================================
if(WIN32)
	set (TARGET_NAME ${TARGET})
else ()
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		set (TARGET_NAME ${TARGET}_zh_d)
	else ()
		set (TARGET_NAME ${TARGET}_zh)
	endif ()
endif()

MESSAGE( STATUS "CMAKE_BUILD_TYPE:	" ${CMAKE_BUILD_TYPE} )
MESSAGE( STATUS "TARGET NAME:		" ${TARGET_NAME} )


#===============================================================================
# Check for machine architecture
#===============================================================================
if (${CMAKE_SIZEOF_VOID_P} EQUAL "8")
    set (PROJECTNAME_ARCHITECTURE "x64")
else ()
    set (PROJECTNAME_ARCHITECTURE "x86")
endif ()

if(WIN32)
	set(CMAKE_GENERATOR_TOOLSET "host=x64")
endif()


#===============================================================================
# Policy for preventing pop-up about warning
#===============================================================================
if(POLICY CMP0012)
  cmake_policy(SET CMP0012 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()


#===============================================================================
# CMake minimum required version (CUDA C++ an intrinsically supported language)
#===============================================================================
if(WIN32)
	cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
endif()


#===============================================================================
# Suppress certain warnings
#===============================================================================

IF(WIN32)
	# With Visual Studio 2005, Microsoft deprecates the standard C library, for
		# example fopen() and sprintf(), to non-portable functions fopen_s() and
		# sprintf_s(). These functions are considered by Microsoft more secure. This is
		# a worthwhile exercise ! The use of these deprecated functions causes a lot of
		# warnings. To suppress it, we add the _CRT_SECURE_NO_DEPRECATE preprocessor
		# definition -fprofile-arcs -ftest-coverage
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS)
endif()


##===============================================================================
# Set compiler flags
##===============================================================================
# Find OpenMP package
find_package(OpenMP)
# MESSAGE( STATUS "OpenMP_CXX_FLAGS:	" ${OpenMP_CXX_FLAGS} )

# Suppress warnings
if (WIN32)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W0")
else()
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")
endif()

# Not making much sensor, but for the GPU system, I have to include compute-version_30
set(CUDA_GENE_CODES "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37")

# Add openmp support and std=c++11 library
if (WIN32)
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /openmp")
else()
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lm -std=c++11 -lpthread -fopenmp -lgomp")
endif()

if (WIN32)
	set(CMAKE_CXX_CUDA_FLAGS "-Xcompiler /openmp ${CUDA_GENE_CODES}")
else()
	set(CMAKE_CXX_CUDA_FLAGS "-lm -std=c++11 -lpthread -Xcompiler -fopenmp -lgomp ${CUDA_GENE_CODES}")
endif()

# MESSAGE( STATUS "CUDA_NVCC_FLAGS:		" ${CUDA_NVCC_FLAGS} )
MESSAGE( STATUS "CMAKE_CXX_FLAGS:	" ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS "CMAKE_CXX_CUDA_FLAGS:	" ${CMAKE_CXX_CUDA_FLAGS} )


#===============================================================================
# Prevent In-Source Build
#===============================================================================
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(SEND_ERROR "In-source builds are not allowed. Create a build directory and try again.")
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})


#===============================================================================
# Setup all directories
#===============================================================================
# ---------- Setup output library Directories -------------------------
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib
   CACHE PATH
   "Directory for all Libraries"
   )

# --------- Setup the Executable output Directory -------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/..
   CACHE PATH
   "Directory for all Executables."
   )

# --------- Setup the Directory for static libraries -------------
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   ${PROJECT_BINARY_DIR}/lib
   CACHE PATH
   "Directory for all static Libraries."
   )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


#===============================================================================
# Whether to copy shared library
#===============================================================================
OPTION(BUILD_SHARED_LIBS "Build shared libraries." ON)


#===============================================================================
# Define home directory for CUDA and Tecplot, needs to change CUDA on new system
#===============================================================================
# SET(TECPLOT_HOME "${PROJECT_SOURCE_DIR}/extern")

IF(WIN32)
	SET(CUDA_HOME "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.1")
ELSE()
	IF(UNIX)
		# One also needs to export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
		SET(CUDA_HOME "/usr/local/cuda")
	ENDIF()
ENDIF()

MESSAGE( STATUS "CUDA_HOME:    " ${CUDA_HOME} )
# MESSAGE( STATUS "TECPLOT_HOME: " ${TECPLOT_HOME} )


#===============================================================================
# Set UM-LISA include directory
#===============================================================================
set(INCLUDE_DIRS
		"${PROJECT_SOURCE_DIR}/src/cpu"
		"${PROJECT_SOURCE_DIR}/src/gpu"
		"${CUDA_HOME}/include"
		)

# include header directory
include_directories(${INCLUDE_DIRS})


#===============================================================================
# CUDA present, add sub directories to execute CMakeLists.txt
#===============================================================================
add_subdirectory(src/cpu)
add_subdirectory(src/gpu)

# Make TARGET default startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})


# #===============================================================================
# # Define CMAKE_INSTALL_PREFIX
# #===============================================================================

# if (WIN32)
	# SET(CMAKE_INSTALL_PREFIX "Z:/${PROJECT_NAME}")
# else()
	# SET(CMAKE_INSTALL_PREFIX "/home/ericzhng")
# endif()
# MESSAGE( STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX} )


#===============================================================================
# Information printing for reference
#===============================================================================
MESSAGE( STATUS "")
MESSAGE( STATUS "PRINT INFORMATION FOR REFERENCE!")
MESSAGE( STATUS "")

MESSAGE( STATUS "PROJECT_SOURCE_DIR:	" ${PROJECT_SOURCE_DIR} )
MESSAGE( STATUS "PROJECT_BINARY_DIR:	" ${PROJECT_BINARY_DIR} )
MESSAGE( STATUS "CMAKE_SOURCE_DIR:      " ${CMAKE_SOURCE_DIR} )
MESSAGE( STATUS "CMAKE_BINARY_DIR:      " ${CMAKE_BINARY_DIR} )

MESSAGE( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )

MESSAGE( STATUS "WIN32: " ${WIN32} )
MESSAGE( STATUS "UNIX:  " ${UNIX} )
MESSAGE( STATUS "APPLE: " ${APPLE} )

