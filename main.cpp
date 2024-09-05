#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>
#pragma comment(lib,"OpenGL32.lib")  

extern "C" int affineTransformDeviceArray(unsigned char* d_input, unsigned char* d_output, int width, int height, float* d_matrix);
extern "C" void affineTransformDevicePBO(GLenum format, GLenum type, unsigned char* d_input, int width, int height, float* d_matrix);

// 纹理ID
GLuint textureID;

// 初始化OpenGL
void initOpenGL(int width, int height) {
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// 设置纹理参数
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// 创建一个空的纹理
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, 0);
}

// 更新纹理数据
void updateTexture(unsigned char* data, int width, int height) {
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// 更新纹理数据
void updateFBOTexture(unsigned char* data, int width, int height) {
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDeleteFramebuffers(1, &fbo);

	//glBindTexture(GL_TEXTURE_2D, textureID);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	//glBindTexture(GL_TEXTURE_2D, 0);
}

// 绘制纹理
void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// 绘制一个四边形
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f); // 左下角
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);  // 右下角
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);   // 右上角
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);  // 左上角
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
}

int main() {
	const int width = 1080;
	const int height = 1080;
	const std::string image_path = "G:/tensorRT_Pro/OutLayerRes/TestOpenGLPro/demo_image.png";
	cv::Mat image = cv::imread(image_path);

	const size_t bgrImageSize = width * height * 3 * sizeof(unsigned char); // BGR
	const size_t rgbaImageSize = width * height * 4 * sizeof(unsigned char); // RGBA

		// Allocate memory for input (BGR) and output (RGBA) images
	unsigned char* h_input = new unsigned char[width * height * 3]; // BGR
	unsigned char* d_input, * d_output;
	float h_matrix[6] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 }; // Identity matrix
	float* d_matrix;

	// Allocate device memory
	cudaMalloc((void**)&d_input, bgrImageSize);
	cudaMalloc((void**)&d_output, rgbaImageSize);
	cudaMalloc((void**)&d_matrix, 6 * sizeof(float));

	// 初始化GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(width, height, "Texture Renderer", nullptr, nullptr);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glewInit();

	// 初始化OpenGL
	initOpenGL(width, height);

	// Copy input data to device
	cudaMemcpy(d_input, image.data, bgrImageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix, h_matrix, 6 * sizeof(float), cudaMemcpyHostToDevice);

	// 主循环
	while (!glfwWindowShouldClose(window)) {
		textureID = affineTransformDeviceArray(d_input, d_output, width, height, d_matrix);
		std::cout << "textureID -------------------------> " << textureID << std::endl;

		display();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// 清理
	// delete[] textureData;
	//cudaGraphicsUnregisterResource(cudaResource);
	glDeleteTextures(1, &textureID);
	glfwDestroyWindow(window);
	glfwTerminate();

	// 清理资源
	glfwTerminate();
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_matrix);
	delete[] h_input;

	return 0;
}

