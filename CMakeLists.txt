cmake_minimum_required(VERSION 3.16)
project(prism VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(prism
    src/main.cpp
)

target_include_directories(prism PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(UNIX AND NOT APPLE)
    target_link_libraries(prism PRIVATE pthread)
endif()
