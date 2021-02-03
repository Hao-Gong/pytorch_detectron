#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision","pycocotools"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir,  "pytorch_detectron","lib_c","model","c_extensions")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="pytorch_detectron",
    version="1.0",
    description="object detection in pytorch",
    author="diamous",
    author_email="diamous@163.com",
    # description="multi detection model",
    # data_files=[('pretrain_models', ['uaesai_detectron/pretrain_models/resnet101.pth',\
    #     'uaesai_detectron/pretrain_models/resnet18.pth','uaesai_detectron/pretrain_models/resnet34.pth',\
    #         'uaesai_detectron/pretrain_models/resnet50.pth','uaesai_detectron/pretrain_models/resnet152.pth',\
    #             'uaesai_detectron/pretrain_models/yoloV3_pretrain.pth','uaesai_detectron/pretrain_models/yolo4_pretrain.pth',\
    #                 'uaesai_detectron/pretrain_models/yolov4_tiny_pretrain.pth','uaesai_detectron/pretrain_models/efficientdet-d0.pth','uaesai_detectron/pretrain_models/efficientdet-d1.pth'])],
    ext_modules=get_extensions(),
    packages=find_packages(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
