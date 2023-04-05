#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 4/2/2023 5:34 AM
# @Author : xia shufan
from setuptools import setup, find_packages
"""
打包的用的setup必须引入，
"""
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='breast-substype-analysis',
    version='1.0.0',
    description='breast-subtype-analysis 是一个用于乳腺癌症分子分型预测的Python包。',
    author='xia shufan',
    author_email='2757462803@qq.com',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
