#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err,
	const char* file, const int line) {
	if (cudaSuccess != err) {
		printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void affineTransformKernel(unsigned char* input, unsigned char* output, int width, int height, float* matrix)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		// Apply the affine transformation
		float newX = matrix[0] * x + matrix[1] * y + matrix[2];
		float newY = matrix[3] * x + matrix[4] * y + matrix[5];

		if (newX >= 0 && newX < width && newY >= 0 && newY < height)
		{
			int bgrIndex = (int)newY * width + (int)newX;
			// Read BGR values
			unsigned char b = input[bgrIndex * 3 + 0];
			unsigned char g = input[bgrIndex * 3 + 1];
			unsigned char r = input[bgrIndex * 3 + 2];

			// Write RGBA values
			int rgbaIndex = (y * width + x) * 4;
			output[rgbaIndex + 0] = r;	 // R
			output[rgbaIndex + 1] = g;	 // G
			output[rgbaIndex + 2] = b;	 // B
			output[rgbaIndex + 3] = 255; // A (fully opaque)
		}
		else
		{
			int rgbaIndex = (y * width + x) * 4;
			output[rgbaIndex + 0] = 0; // R
			output[rgbaIndex + 1] = 0; // G
			output[rgbaIndex + 2] = 0; // B
			output[rgbaIndex + 3] = 0; // A (transparent)
		}
	}
}

extern "C" int affineTransformDeviceArray(unsigned char* d_input, unsigned char* d_output, int width, int height, float* d_matrix)
{


	// 初始化两个Texture并绑定
	cudaGraphicsResource* cudaResources;
	GLuint textureID;
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// 在CUDA中注册Texture
	cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResources, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess)
	{
		std::cout << "cudaGraphicsGLRegisterImage: " << err << "Line: " << __LINE__;
		return -1;
	}

	// 在CUDA中锁定资源，获得操作Texture的指针，这里是CudaArray*类型
	err = cudaGraphicsMapResources(1, &cudaResources, 0);
	cudaArray_t cuArray;
	err = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResources, 0, 0);
	if (err != cudaSuccess)
	{
		std::cout << "cudaGraphicsSubResourceGetMappedArray: " << err << "Line: " << __LINE__;
		return -1;
	}

	//unsigned char* dstPointer;
	//size_t size;
	//err = cudaGraphicsResourceGetMappedPointer((void**)&dstPointer, &size, cudaResources);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	affineTransformKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, d_matrix);
	cudaDeviceSynchronize();

	// 数据拷贝至CudaArray。这里因为得到的是CudaArray，处理时不方便操作，于是先在设备内存中
	// 分配缓冲区处理，处理完后再把结果存到CudaArray中，仅仅是GPU内存中的操作。
	cudaMemcpyToArray(cuArray, 0, 0, (void*)d_output, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice);
	// cudaMemcpy(dstPointer, d_output, width * height * 4, cudaMemcpyDeviceToDevice);
	// 处理完后即解除资源锁定，OpenGL可以利用得到的Texture对象进行纹理贴图操作了。
	cudaGraphicsUnmapResources(1, &cudaResources, 0);

	//// Cleanup the resource
	cudaGraphicsUnregisterResource(cudaResources);
	//cudaFree(d_output); // Free the output memory

	return static_cast<int>(textureID);
}


extern "C" void affineTransformDevicePBO(GLenum format, GLenum type, unsigned char* d_input, int width, int height, float* d_matrix)
{


	// 初始化两个Texture并绑定
	cudaGraphicsResource* cudaResources;
	GLuint pbo;
	void* device_ptr = nullptr;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * sizeof(unsigned) * height, 0, GL_STREAM_COPY);
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cudaResources, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
	glReadPixels(0, 0, width, height, format, type, 0);

	cudaGraphicsMapResources(1, &cudaResources);
	size_t size = 0;
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, cudaResources));
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	affineTransformKernel << <gridSize, blockSize >> > (d_input, (unsigned char*)device_ptr, width, height, d_matrix);
	cudaDeviceSynchronize();

	// 处理完后即解除资源锁定，OpenGL可以利用得到的Texture对象进行纹理贴图操作了。
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cudaResources));

	//// Cleanup the resource
	cutilSafeCall(cudaGraphicsUnregisterResource(cudaResources));
	glDeleteBuffers(1, &pbo);

}
