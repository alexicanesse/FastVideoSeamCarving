cmake_minimum_required (VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast")


project ("seamCarving")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

###############################################################################
## file globbing ##############################################################
###############################################################################

set(CXXFILES ./src/seamCarving.cpp)

###############################################################################
## target definitions #########################################################
###############################################################################

include_directories("./include")
add_executable(seamCarving.out  ${CXXFILES})


###############################################################################
## dependencies ###############################################################
###############################################################################

# Find the OpenCV library
find_package(OpenCV REQUIRED)

# Link against the OpenCV library
target_link_libraries(seamCarving.out ${OpenCV_LIBS})