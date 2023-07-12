#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include "BasicBlock.hpp"
#include "bottleneck.hpp"


template<typename Block>
class ResNetImpl : public torch::nn::Module {
public:
    explicit ResNetImpl(const std::vector<int64_t> layers, int64_t num_classes = 1000,
                        bool zero_init_residual = false, int64_t groups = 1,
                        int64_t width_per_group = 64,
                        std::vector<int64_t> replace_stride_with_dilation = {}) {
        if (replace_stride_with_dilation.size() == 0) {
            // Each element in the tuple indicates if we should replace
            // the 2x2 stride with a dilated convolution instead.
            replace_stride_with_dilation = {false, false, false};
        }
        if (replace_stride_with_dilation.size() != 3) {
            throw std::invalid_argument{
                    "replace_stride_with_dilation should be empty or have exactly "
                    "three elements."};
        }

        m_groups = groups;
        m_base_width = width_per_group;

        m_conv1 = register_module(
                "conv1",
                torch::nn::Conv2d{
                        torch::nn::Conv2dOptions(/* in_planes =*/ 3,/* out_planes = */ m_inplanes, /*kerner_size = */ 7)
                                .stride(2)
                                .padding(3)
                                .groups(1)
                                .dilation(1)
                                .bias(false)
//                    create_conv_options(
//                        /*in_planes = */ 3, /*out_planes = */ m_inplanes,
//                        /*kerner_size = */ 7, /*stride = */ 2, /*padding = */ 3,
//                        /*groups = */ 1, /*dilation = */ 1, /*bias = */ false)
                });
        std::cout << "resnet m_conv1:" << m_conv1 << '\n';
        m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{m_inplanes});
        std::cout << "resnet m_bn1:" << m_bn1 << '\n';
        m_relu = register_module("relu", torch::nn::ReLU{true});
        m_maxpool = register_module(
                "maxpool",
                torch::nn::MaxPool2d{
                        torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding(
                                {1, 1})});

        m_layer1 = register_module("layer1", _make_layer(64, layers.at(0)));
        m_layer2 = register_module(
                "layer2", _make_layer(128, layers.at(1), 2,
                                      replace_stride_with_dilation.at(0)));
        m_layer3 = register_module(
                "layer3", _make_layer(256, layers.at(2), 2,
                                      replace_stride_with_dilation.at(1)));
        m_layer4 = register_module(
                "layer4", _make_layer(512, layers.at(3), 2,
                                      replace_stride_with_dilation.at(2)));

        m_avgpool = register_module(
                "avgpool", torch::nn::AdaptiveAvgPool2d(
                        torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        m_fc = register_module(
                "fc", torch::nn::Linear(512 * Block::Impl::m_expansion, num_classes));

        // auto all_modules = modules(false);
        // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
        for (auto m: modules(false)) {
            if (m->name() == "torch::nn::Conv2dImpl") {
                torch::OrderedDict<std::string, torch::Tensor>
                        named_parameters = m->named_parameters(false);
                torch::Tensor *ptr_w = named_parameters.find("weight");
                torch::nn::init::kaiming_normal_(*ptr_w, 0, torch::kFanOut,
                                                 torch::kReLU);
            } else if ((m->name() == "torch::nn::BatchNormImpl") ||
                       (m->name() == "torch::nn::GroupNormImpl")) {
                torch::OrderedDict<std::string, torch::Tensor>
                        named_parameters = m->named_parameters(false);
                torch::Tensor *ptr_w = named_parameters.find("weight");
                torch::nn::init::constant_(*ptr_w, 1.0);
                torch::Tensor *ptr_b = named_parameters.find("bias");
                torch::nn::init::constant_(*ptr_b, 0.0);
            }
        }

        if (zero_init_residual) {
            for (auto m: modules(false)) {
                if (m->name() == "Bottleneck") {
                    torch::OrderedDict<std::string, torch::Tensor>
                            named_parameters =
                            m->named_modules()["bn3"]->named_parameters(false);
                    torch::Tensor *ptr_w = named_parameters.find("weight");
                    torch::nn::init::constant_(*ptr_w, 0.0);
                } else if (m->name() == "BasicBlock") {
                    torch::OrderedDict<std::string, torch::Tensor>
                            named_parameters =
                            m->named_modules()["bn2"]->named_parameters(false);
                    torch::Tensor *ptr_w = named_parameters.find("weight");
                    torch::nn::init::constant_(*ptr_w, 0.0);
                }
            }
        }
    }

    int64_t m_inplanes = 64;
    int64_t m_dilation = 1;
    int64_t m_groups = 1;
    int64_t m_base_width = 64;

    torch::nn::Conv2d m_conv1{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::MaxPool2d m_maxpool{nullptr};
    torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr},
            m_layer3{nullptr}, m_layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
    torch::nn::Linear m_fc{nullptr};

    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
                                      int64_t stride = 1, bool dilate = false) {
        torch::nn::Sequential downsample = torch::nn::Sequential();
        int64_t previous_dilation = m_dilation;
        if (dilate) {
            m_dilation *= stride;
            stride = 1;
        }
        if ((stride != 1) || (m_inplanes != planes * Block::Impl::m_expansion)) {
            downsample = torch::nn::Sequential(
                    conv1x1(m_inplanes, planes * Block::Impl::m_expansion, stride),
                    torch::nn::BatchNorm2d(planes * Block::Impl::m_expansion));
        }

        torch::nn::Sequential layers;

        layers->push_back(Block(m_inplanes, planes, stride, downsample,
                                m_groups, m_base_width, previous_dilation));
        m_inplanes = planes * Block::Impl::m_expansion;
        for (int64_t i = 0; i < blocks; i++) {
            layers->push_back(Block(m_inplanes, planes, 1,
                                    torch::nn::Sequential(), m_groups,
                                    m_base_width, m_dilation));
        }

        return layers;
    }

    torch::Tensor _forward_impl(torch::Tensor x) {

        x = m_conv1->forward(x);
        x = m_bn1->forward(x);
        x = m_relu->forward(x);
        x = m_maxpool->forward(x);

        x = m_layer1->forward(x);
        x = m_layer2->forward(x);
        x = m_layer3->forward(x);
        x = m_layer4->forward(x);

        x = m_avgpool->forward(x);
        x = torch::flatten(x, 1);
        x = m_fc->forward(x);

        return x;
    }

    torch::Tensor forward(torch::Tensor x) { return _forward_impl(x); }
};

template<typename T>
TORCH_MODULE_IMPL(ResNet, ResNetImpl<T>);
//template<typename T>
//class ResNet : public torch::nn::ModuleHolder<ResNetImpl<T> > {
//public:
//    using torch::nn::ModuleHolder<ResNetImpl<T> >::ModuleHolder;
//    using Impl TORCH_UNUSED_EXCEPT_CUDA = ResNetImpl<T>;
//};

