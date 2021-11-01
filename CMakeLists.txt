cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(EXECUTABLE_NAME tensor-experiment)

#set(CMAKE_PREFIX_PATH /home/weizhen/Documents/libtorch)
#set(Torch_DIR /home/PROJECTS/libtorch/libtorch/share/cmake/Torch)
set(Torch_DIR /home/PROJECTS/Landmarks/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${EXECUTABLE_NAME} 
				main.cpp 
				autoencoder.cpp
				CustomDataset.cpp
				CSVLoader.cpp)
				
target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 14)
