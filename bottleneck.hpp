#pragma once

#include <torch/torch.h>
#include "defines.hpp"


class BottleneckImpl : public torch::nn::Module {
public:
    BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride = 1,
                   torch::nn::Sequential downsample = torch::nn::Sequential(),
                   int64_t groups = 1, int64_t base_width = 64,
                   int64_t dilation = 1) {
        int64_t width = planes * (base_width / 64) * groups;

        m_conv1 = register_module(
                "conv1", conv1x1(inplanes, width));
        m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{width});
        m_conv2 = register_module("conv2", conv3x3(
                width, width, stride, groups, dilation));
        m_bn2 = register_module("bn2", torch::nn::BatchNorm2d{width});
        m_conv3 =
                register_module("conv3", conv1x1(
                        width, planes * m_expansion));
        m_bn3 = register_module("bn3",
                                torch::nn::BatchNorm2d{planes * m_expansion});
        m_relu = register_module("relu", torch::nn::ReLU{true});
        if (!downsample->is_empty()) {
            m_downsample = register_module("downsample", downsample);
        }
        m_stride = stride;
    }

    static const int64_t m_expansion = 4;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr}, m_conv3{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr}, m_bn3{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Sequential m_downsample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(const torch::Tensor &x) {
        torch::Tensor identity = x;

        torch::Tensor out = m_conv1->forward(x);
        out = m_bn1->forward(out);
        out = m_relu->forward(out);

        out = m_conv2->forward(out);
        out = m_bn2->forward(out);
        out = m_relu->forward(out);

        out = m_conv3->forward(out);
        out = m_bn3->forward(out);

        if (!m_downsample->is_empty()) {
            identity = m_downsample->forward(x);
        }

        out += identity;
        out = m_relu->forward(out);

        return out;
    }
};

TORCH_MODULE(Bottleneck);