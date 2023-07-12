#pragma once
#include <torch/torch.h>


auto conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride = 1, int64_t groups = 1, int64_t dilation = 1) {
    return torch::nn::Conv2d{
            torch::nn::Conv2dOptions(in_planes, out_planes, 3)
                    .stride(stride)
                    .padding(dilation)
                    .groups(groups)
                    .dilation(dilation)
                    .bias(false)
    };
}

auto conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride = 1) {
    return torch::nn::Conv2d{
            torch::nn::Conv2dOptions(in_planes, out_planes, 1)
                    .stride(stride)
                    .bias(false)
    };
}