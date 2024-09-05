cmake_minimum_required(VERSION 3.10)
project(AffineTransformCUDAOpenGL)

find_package(CUDA REQUIRED)

set(GLFW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/opengl/include)
set(GLFW_LIBRARIES ${PROJECT_SOURCE_DIR}/opengl/lib/x64)

set(GLM_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/opengl/include/glm)

set(OPENGL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/opengl/include)
set(OPENGL_LIBRARIES ${PROJECT_SOURCE_DIR}/opengl/lib/x64)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 14)

cuda_add_library(affine_transform
    affine_transform.cu
)
message("OPENGL info ---------> ${OPENGL_LIBRARIES}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE affine_transform ${OPENGL_LIBRARIES}/glew32s.lib ${GLFW_LIBRARIES}/glfw3.lib ${GLFW_LIBRARIES}/freeglut_static.lib)


# cmake_minimum_required(VERSION 3.10)
# project(AffineTransformCUDAOpenGL)

# find_package(CUDA REQUIRED)
# # find_package(OpenGL REQUIRED)
# # find_package(GLM REQUIRED)
# set(GLFW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/glfw/include)
# set(GLFW_LIBRARIES ${PROJECT_SOURCE_DIR}/glfw/out/win64)

# set(GLM_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/glm)

# set(OPENGL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/glew-2.1.0/include)
# set(OPENGL_LIBRARIES ${PROJECT_SOURCE_DIR}/glew-2.1.0/lib/Release/x64)
# include_directories(${OPENGL_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

# set(CMAKE_CUDA_STANDARD 14)

# cuda_add_library(affine_transform
#     affine_transform.cu
# )
# message("OPENGL info ---------> ${OPENGL_LIBRARIES}")

# add_executable(main main.cpp)
# target_link_libraries(main PRIVATE affine_transform ${OPENGL_LIBRARIES}/glew32s.lib ${GLFW_LIBRARIES}/glfw3.lib)
