cmake_minimum_required(VERSION 2.8.12)
project(pfnncharacter)

# -Wno-deprecated -Wno-macro-redefined 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
message(${CMAKE_CXX_FLAGS})

set(Python3_ROOT_DIR "")
find_package (Python3 3.7 EXACT COMPONENTS Interpreter Development)

message(${Python3_FOUND})
message(${Python3_EXECUTABLE})
message(${Python3_INCLUDE_DIRS})
message(${Python3_LIBRARIES})
message(${Python3_LIBRARY_DIRS})


add_subdirectory(pybind11)
add_subdirectory(glm)
#add_subdirectory(eigen-3.3.7)

include_directories(${Python3_INCLUDE_DIRS} pybind11/include glm eigen-3.3.7)


option(BUILD_AS_EXECUTABLE "Enable to build as executable, or off to build as a shared lib" OFF)
if (BUILD_AS_EXECUTABLE)
    add_executable(pfnncharacter main.cpp PFNNBaseCode.cpp)
    target_link_libraries(pfnncharacter ${Python3_LIBRARIES})
    message("building as executable")
    add_definitions(-DBUILD_AS_EXECUTABLE)
else()
    pybind11_add_module(pfnncharacter main.cpp "PFNNBaseCode.cpp")
    message("building as shared lib")
endif()
