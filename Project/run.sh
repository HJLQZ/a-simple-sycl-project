#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling - 1 of 1 main.cpp
icpx -fsycl main.cpp 
if [ $? -eq 0 ]; then ./a.out; fi