cmake_minimum_required(VERSION 3.10)
project(HogSvmCpp)

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")

# Create Main project
file(GLOB SOURCES src/*.cpp)
add_executable(Main ${SOURCES})
target_include_directories(Main PUBLIC include)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(Main PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(Main ${OpenCV_LIBS})
endif()

set_property(TARGET ${ProjectName} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${ProjectName} PROPERTY CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)