#!/bin/bash

set -e

if [[ $(uname) == "Darwin" ]]; then
    OS=mac
else
    OS=linux
fi

GLFW_HOME=$(cd $(dirname $0)/..; pwd)
BUILD_DIR=${GLFW_HOME}/build_${OS}
OUTPUT_DIR=${GLFW_HOME}/out/${OS}

[[ -d ${BUILD_DIR} ]] || mkdir ${BUILD_DIR}

pushd ${BUILD_DIR}

cmake ${GLFW_HOME}
make -j$(nproc)

[[ -d ${OUTPUT_DIR} ]] || mkdir -p ${OUTPUT_DIR}
cp ${BUILD_DIR}/src/libglfw3.a ${OUTPUT_DIR}

popd
