cmake_minimum_required(VERSION 3.20)
project(cifar-demo-test CXX)

# need torch
find_package(Torch REQUIRED)

add_executable(
        ${PROJECT_NAME}
        resnet-cifar-demo.cpp
        cifar10.hpp
)
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/../out/${CMAKE_BUILD_TYPE}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        "${TORCH_LIBRARIES}"
        "resnet"
)

if (MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif ()
