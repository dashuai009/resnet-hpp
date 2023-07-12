#pragma once
#include "resnet.hpp"

namespace resnet_space {
template<class Block>
ResNet<Block> _resnet(const std::vector<int64_t> &layers, int64_t num_classes = 1000, bool zero_init_residual = false,
                      int64_t groups = 1, int64_t width_per_group = 64,
                      const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return ResNet<Block>(layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation);
}

ResNet<BasicBlock>
resnet18(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return _resnet<BasicBlock>({2, 2, 2, 2}, num_classes, zero_init_residual, groups, width_per_group,
                               replace_stride_with_dilation);
}

ResNet<BasicBlock>
resnet34(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return _resnet<BasicBlock>({3, 4, 6, 3}, num_classes, zero_init_residual, groups,
                               width_per_group, replace_stride_with_dilation);
}

ResNet<Bottleneck>
resnet50(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return _resnet<Bottleneck>({3, 4, 6, 3}, num_classes, zero_init_residual, groups,
                               width_per_group, replace_stride_with_dilation);
}

ResNet<Bottleneck>
resnet101(int64_t num_classes = 1000, bool zero_init_residual = false,
          int64_t groups = 1, int64_t width_per_group = 64,
          const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return _resnet<Bottleneck>({3, 4, 23, 3}, num_classes, zero_init_residual, groups,
                               width_per_group, replace_stride_with_dilation);
}

ResNet<Bottleneck>
resnet152(int64_t num_classes = 1000, bool zero_init_residual = false,
          int64_t groups = 1, int64_t width_per_group = 64,
          const std::vector<int64_t> &replace_stride_with_dilation = {}) {
    return _resnet<Bottleneck>({3, 8, 36, 3}, num_classes, zero_init_residual, groups,
                               width_per_group, replace_stride_with_dilation);
}

ResNet<Bottleneck>
resnext50_32x4d(int64_t num_classes = 1000, bool zero_init_residual = false,
                int64_t groups = 1, int64_t width_per_group = 64,
                const std::vector<int64_t>& replace_stride_with_dilation = {}) {
    groups = 32;
    width_per_group = 4;
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<Bottleneck> model =
            _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck>
resnext101_32x8d(int64_t num_classes = 1000, bool zero_init_residual = false,
                 int64_t groups = 1, int64_t width_per_group = 64,
                 std::vector<int64_t> replace_stride_with_dilation = {}) {
    groups = 32;
    width_per_group = 8;
    const std::vector<int64_t> layers{3, 4, 23, 3};
    ResNet<Bottleneck> model =
            _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck>
wide_resnet50_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                int64_t groups = 1, int64_t width_per_group = 64,
                std::vector<int64_t> replace_stride_with_dilation = {}) {
    width_per_group = 64 * 2;
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<Bottleneck> model =
            _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck>
wide_resnet101_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                 int64_t groups = 1, int64_t width_per_group = 64,
                 std::vector<int64_t> replace_stride_with_dilation = {}) {
    width_per_group = 64 * 2;
    const std::vector<int64_t> layers{3, 4, 23, 3};
    ResNet<Bottleneck> model =
            _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                width_per_group, replace_stride_with_dilation);
    return model;
}

template
struct ResNet<BasicBlock>;
template
struct ResNet<Bottleneck>;

}// resnet_sapce