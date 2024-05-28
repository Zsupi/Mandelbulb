#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "interop.cuh"
#include "stdio.h"
#include "camera/perspective_camera.h"
#include <memory>
#include <cuda/std/complex>
#include <glfwim/input_manager.h>

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

uint32_t imageWidth, imageHeight;
cudaExternalMemory_t cudaExtMemImageBuffer; // memory handler to the imported memory allocation
cudaMipmappedArray_t cudaMipmappedImageArray; // the image interpreted as a mipmapped array
cudaSurfaceObject_t surfaceObject; // surface object to the first mip level of the array. Allows write

struct Camera {
    glm::vec3 pos;
    glm::vec3 dir;
    glm::vec3 up;
    glm::vec3 right;
    float hFov;
};

void freeExportedVulkanImage()
{
    checkCudaError(cudaDestroySurfaceObject(surfaceObject));
    checkCudaError(cudaFreeMipmappedArray(cudaMipmappedImageArray));
    checkCudaError(cudaDestroyExternalMemory(cudaExtMemImageBuffer));
}

void exportVulkanImageToCuda_R8G8B8A8Unorm(void* mem, VkDeviceSize size, VkDeviceSize offset, uint32_t width, uint32_t height)
{
    imageWidth = width;
    imageHeight = height;

    // import memory into cuda through native handle (win32)
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
    memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
    cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32; // after win8
    cudaExtMemHandleDesc.handle.win32.handle = mem; // allocation handle
    cudaExtMemHandleDesc.size = size; // allocation size
   
    checkCudaError(cudaImportExternalMemory(&cudaExtMemImageBuffer, &cudaExtMemHandleDesc));

    // extract mipmapped array from memory
    cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;
    memset(&externalMemoryMipmappedArrayDesc, 0, sizeof(externalMemoryMipmappedArrayDesc));

    // we want ot interpret the raw memory as an image so we need to specify its format and layout
    cudaExtent extent = make_cudaExtent(width, height, 0);
    cudaChannelFormatDesc formatDesc; // 4 channel, 8 bit per channel, unsigned
    formatDesc.x = 8;
    formatDesc.y = 8;
    formatDesc.z = 8;
    formatDesc.w = 8;
    formatDesc.f = cudaChannelFormatKindUnsigned;

    externalMemoryMipmappedArrayDesc.offset = offset; // the image starts here
    externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
    externalMemoryMipmappedArrayDesc.extent = extent;
    externalMemoryMipmappedArrayDesc.flags = 0;
    externalMemoryMipmappedArrayDesc.numLevels = 1; // no mipmapping
    checkCudaError(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArray, cudaExtMemImageBuffer, &externalMemoryMipmappedArrayDesc));

    // extract first level
    cudaArray_t cudaMipLevelArray;
    checkCudaError(cudaGetMipmappedArrayLevel(&cudaMipLevelArray, cudaMipmappedImageArray, 0));

    // create surface object for writing
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaMipLevelArray;
    
    checkCudaError(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
}

cudaExternalSemaphore_t cudaWaitsForVulkanSemaphore, vulkanWaitsForCudaSemaphore;

void freeExportedSemaphores()
{
    checkCudaError(cudaDestroyExternalSemaphore(cudaWaitsForVulkanSemaphore));
    checkCudaError(cudaDestroyExternalSemaphore(vulkanWaitsForCudaSemaphore));
}

void exportSemaphoresToCuda(void* cudaWaitsForVulkanSemaphoreHandle, void* vulkanWaitsForCudaSemaphoreHandle) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.flags = 0;
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

    externalSemaphoreHandleDesc.handle.win32.handle = cudaWaitsForVulkanSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&cudaWaitsForVulkanSemaphore, &externalSemaphoreHandleDesc));

    externalSemaphoreHandleDesc.handle.win32.handle = vulkanWaitsForCudaSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&vulkanWaitsForCudaSemaphore, &externalSemaphoreHandleDesc));
}

// compresses 4 32bit floats into a 32bit uint
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
        ((unsigned int)(rgba.z * 255.0f) << 16) |
        ((unsigned int)(rgba.y * 255.0f) << 8) |
        ((unsigned int)(rgba.x * 255.0f));
}

__device__ float map(float x, float fromMin, float fromMax, float toMin, float toMax) {
    return toMin + (toMax - toMin) * (x - fromMin) / (fromMax - fromMin);
}

__device__ float sdSphere(glm::vec3 p)
{
    return length(p) - 0.25f;
}

__device__ float2 sdMandelbrot(glm::vec3 p, int maxIter) {
    // article: https://iquilezles.org/articles/mandelbulb/
    glm::vec3 w = p;
    float power = 8.0f;
    float m = dot(w,w);

    float dz = 1.0f;

    int i;

    for( i=0; i<maxIter; i++ ) {
        dz = power * pow(m,3.5f)*dz + 1.0f;

        // extract polar coordinates
        float wr = sqrt(dot(w, w));
        float wo = acos(w.y / wr);
        float wi = glm::atan(w.x, w.z);

        // scale and rotate the point
        wr = pow(wr, power);
        wo = wo * power;
        wi = wi * power;

        // convert back to cartesian coordinates
        w.x = wr * sin(wo) * sin(wi);
        w.y = wr * cos(wo);
        w.z = wr * sin(wo) * cos(wi);

        // add the current point to the cartesian coordinates
        w += p;

        // calculate the squared distance
        m = dot(w,w);

        // if the distance is greater than 256, break
        if( m > 32.0f )
            break;
    }

    float dist = 0.25f*log(m)*sqrt(m)/dz;
    return float2 { dist, float(i) };
}

__device__ glm::vec3 colorByDistanceAndIteration(glm::vec3 pos, float maxDist, int iterations, int maxIterations) {
    float distance = length(pos);
    float distanceFactor = distance / maxDist * 3.5f;
    float iterationFactor = float(iterations) / float(maxIterations);

    glm::vec3 distanceColor = glm::vec3(0.45f, 0.82f, 0.94f) * distanceFactor;
    glm::vec3 iterationColor = glm::vec3(0.05f, 1.0f, 0.0f) * iterationFactor;

    return mix(distanceColor, iterationColor, 0.45f);
}

__device__ glm::vec3 calcNormal(glm::vec3 p, int maxIterations) {
    float e = 0.001f;
    return glm::normalize(glm::vec3(
            sdMandelbrot(p + glm::vec3(e, 0, 0), maxIterations).x - sdMandelbrot(p - glm::vec3(e, 0, 0), maxIterations).x,
            sdMandelbrot(p + glm::vec3(0, e, 0), maxIterations).x - sdMandelbrot(p - glm::vec3(0, e, 0), maxIterations).x,
            sdMandelbrot(p + glm::vec3(0, 0, e), maxIterations).x - sdMandelbrot(p - glm::vec3(0, 0, e), maxIterations).x
    ));
}

__device__ glm::vec3 shade(glm::vec3 normal, glm::vec3 pos, glm::vec3 hit, glm::vec3 color) {
    float ks = 0.8f;
    float shininess = 100.0f;
    glm::vec3 powerDensity = glm::vec3(3.0f, 3.0f, 3.0f);
    glm::vec3 lightPos = glm::vec3(30.0f, 30.0f, -30.0f);

    glm::vec3 lightDir = normalize(lightPos - hit);
    glm::vec3 viewDir = normalize(hit - pos);

    float cosa = glm::clamp( dot(lightDir, normal), 0.0f, 1.0f);

    return powerDensity * cosa * color + powerDensity * pow(glm::clamp(dot(normalize(viewDir + lightDir), normal), 0.0f, 1.0f), shininess) * ks;
}

__device__ float4 trace(glm::vec3 pos, glm::vec3 dir) {

    glm::vec3 e = pos;
    glm::vec3 d = glm::normalize(dir);

    int maxIterations = 8;
    float maxDist = 5.0f;

    for (int i = 0; i < 100; i++) {
        float2 result = sdMandelbrot(e, maxIterations);
        float dist = result.x;
        int iterations = int(result.y);
        if (dist < 0.001f) {
            glm::vec3 baseColor = colorByDistanceAndIteration(e, maxDist, iterations, maxIterations);
            glm::vec3 normal = calcNormal(e, int(float(maxIterations) / 2.0f));
            glm::vec3 color = shade(normal, pos, e, baseColor);
            return float4{ color.x, color.y, color.z, 1.0f };
        }
        if (dist > maxDist) break;
        e = e + d * dist;
    }

    return float4{ 0.0f, 0.0f, 0.0f, 1.0f };
}

__global__ void renderToSurface(cudaSurfaceObject_t dstSurface, size_t width, size_t height, Camera camera) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 dataOut;

    float wx = (float(x) / float(width) - 0.5f) * 2.0f;
    float wy = - (float(y) / float(height) - 0.5f) * 2.0f;
    wy *= float(height) / float(width);

    glm::vec3 rayDir = glm::normalize(camera.dir + camera.right * wx + camera.up * wy);
    dataOut = trace(camera.pos, rayDir);

    surf2Dwrite(rgbaFloatToInt(dataOut), dstSurface, x * 4, y);
}

void cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore) {
    cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
    memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

    extSemaphoreSignalParams.params.fence.value = 0;
    extSemaphoreSignalParams.flags = 0;
    checkCudaError(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemaphoreSignalParams, 1));
}

void cudaVkSemaphoreWait(cudaExternalSemaphore_t& extSemaphore) {
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;

    memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));

    extSemaphoreWaitParams.params.fence.value = 0;
    extSemaphoreWaitParams.flags = 0;

    checkCudaError(cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemaphoreWaitParams, 1));
}

bool firstRun = true;

void renderCuda(float zoom, glm::vec2 cursorPos, PerspectiveCamera camera)
{
    cudaVkSemaphoreSignal(vulkanWaitsForCudaSemaphore);
    if (!firstRun) {
        cudaVkSemaphoreWait(cudaWaitsForVulkanSemaphore);
    } else {
        firstRun = false;
    }

    uint32_t nthreads = 32;
    dim3 dimBlock{ nthreads, nthreads };
    dim3 dimGrid{ imageWidth / nthreads + 1, imageHeight / nthreads + 1 };

    Camera cudaCamera = Camera();
    cudaCamera.pos = camera.eyePos();
    cudaCamera.dir = camera.dir();
    cudaCamera.up = camera.Up();
    cudaCamera.right = camera.Right();
    cudaCamera.hFov = camera.fov();

    // Call the renderMandelbulb function
    renderToSurface<<<dimGrid, dimBlock>>>(surfaceObject, imageWidth, imageHeight, cudaCamera);
    checkCudaError(cudaGetLastError());

    //checkCudaError(cudaDeviceSynchronize()); // not optimal! should be synced with vulkan using semaphores

    cudaVkSemaphoreSignal(cudaWaitsForVulkanSemaphore);
    cudaVkSemaphoreWait(vulkanWaitsForCudaSemaphore);
}