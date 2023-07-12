#pragma once

#include <torch/torch.h>
#include "defines.hpp"

class BasicBlockImpl : public torch::nn::Module {
public:
    BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride = 1,
                   torch::nn::Sequential downsample = torch::nn::Sequential(),
                   int64_t groups = 1, int64_t base_width = 64,
                   int64_t dilation = 1) {
        if ((groups != 1) || (base_width != 64)) {
            throw std::invalid_argument{
                    "BasicBlock only supports groups=1 and base_width=64"};
        }
        if (dilation > 1) {
            throw std::invalid_argument{
                    "Dilation > 1 not supported in BasicBlock"};
        }
        m_conv1 = register_module(
                "conv1",
                conv3x3(in_planes, planes, stride)
        );
        m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{planes});
        m_relu = register_module("relu", torch::nn::ReLU{true});
        m_conv2 = register_module(
                "conv2", torch::nn::Conv2d{conv3x3(planes, planes)});
        m_bn2 = register_module("bn2", torch::nn::BatchNorm2d{planes});
        if (!downsample->is_empty()) {
            m_downsample = register_module("downsample", downsample);
        }
        m_stride = stride;
    }


    static const int64_t m_expansion = 1;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr};
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

        if (!m_downsample->is_empty()) {
            identity = m_downsample->forward(x);
        }

        out += identity;
        out = m_relu->forward(out);

        return out;
    }
};

TORCH_MODULE(BasicBlock);