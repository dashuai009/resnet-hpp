cmake_minimum_required(VERSION 3.20)

project(resnet_lib CXX)

find_package(Torch REQUIRED)
message(STATUS "Torch version ${Torch_VERSION}")

## resnet interface begin
add_library(resnet INTERFACE
        BasicBlock.hpp
        bottleneck.hpp
        defines.hpp
        resnet.hpp
        impl.hpp
        )
target_link_libraries(resnet INTERFACE ${TORCH_LIBRARIES})
set_target_properties(resnet
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        LINKER_LANGUAGE CXX
        )
## resnet end

add_subdirectory(test)