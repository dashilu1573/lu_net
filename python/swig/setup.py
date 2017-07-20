#!/usr/bin/env python
#-*- coding: utf-8 -*-

# python自带一个distutils工具，可以用它来创建python的扩展模块。
from distutils.core import setup, Extension

# 生成一个扩展模块
net_module = Extension('_net',  # 模块名称，必须要有下划线
                       sources=['net_wrap.cxx',
                                '../../src/net.cpp',
                                ],
                       )

setup(name='net',   # 打包后的名称
      version='0.1',
      author='SWIG Docs',
      description='Simple swig example from docs',
      ext_modules=[net_module],  # 与上面的扩展模块名称一致
      py_modules=['net'],   # 需要打包的模块列表
    )
