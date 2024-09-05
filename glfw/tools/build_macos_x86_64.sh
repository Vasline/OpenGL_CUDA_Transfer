#!/bin/bash

set -e

OS=mac

GLFW_HOME=$(cd $(dirname $0)/..; pwd)
BUILD_DIR=${GLFW_HOME}/build_${OS}_x86_64
OUTPUT_DIR=${GLFW_HOME}/out/${OS}_x86_64

[[ -d ${BUILD_DIR} ]] || mkdir ${BUILD_DIR}

pushd ${BUILD_DIR}

cmake ${GLFW_HOME} -DCMAKE_OSX_DEPLOYMENT_TARGET=10.10 \
    -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
    -DCMAKE_OSX_ARCHITECTURES=x86_64
make -j$(nproc)

[[ -d ${OUTPUT_DIR} ]] || mkdir -p ${OUTPUT_DIR}
cp ${BUILD_DIR}/src/libglfw3.a ${OUTPUT_DIR}

popd
