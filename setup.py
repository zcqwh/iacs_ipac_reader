#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
      packages=setuptools.find_namespace_packages(
                     include=["iacs_ipac_reader", "iacs_ipac_reader.*"], ),
                 include_package_data=True
                 )