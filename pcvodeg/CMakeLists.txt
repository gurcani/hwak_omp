cmake_minimum_required(VERSION 3.10)
# set the project name
project(pcvodeg VERSION 0.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED)

message([STATUS] ${LIBRARY_DIRS})

find_library(SUNDIALS_SOLVER_LIB sundials_cvode HINT "/linkhome/rech/genlpp01/ucj34vp/local/sundials/lib64" DOC "CVODE library" )
find_library(SUNDIALS_NVECOPENMP_LIB sundials_nvecopenmp HINT "/linkhome/rech/genlpp01/ucj34vp/local/sundials/lib64" DOC "NVECTOR_OPENMP library" )

set(SUNDIALS_LIBRARIES
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVECOPENMP_LIB})

# add_definitions(-DOMPI_SKIP_MPICXX)
# include_directories(${MPI_INCLUDE_PATH})
add_library (${PROJECT_NAME} SHARED ${PROJECT_NAME}.c)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${SUNDIALS_LIBRARIES})
