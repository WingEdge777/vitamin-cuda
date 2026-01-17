#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>

inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                       {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                       {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                       {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xa1, 128},
                                       {0xa3, 128}, {0xb0, 128}, {0xc0, 128}, {0xc1, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf("MapSMtoCores for SM %d.%d is undefined."
           "  Default to use %d Cores/SM\n",
           major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int main(int argc, char **argv) {
    printf("%s running...\n\n", argv[0]);
    int device_cnt = 0;
    cudaGetDeviceCount(&device_cnt);
    printf(":: %d available devices\n\n", device_cnt);
    int drv_ver, runtime_ver;
    for (int dev = 0; dev < device_cnt; dev++) {
        cudaSetDevice(dev);
        cudaDeviceProp device_prop;

        cudaDriverGetVersion(&drv_ver);
        cudaRuntimeGetVersion(&runtime_ver);
        cudaGetDeviceProperties(&device_prop, dev);

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", drv_ver / 1000, (drv_ver % 100) / 10,
               runtime_ver / 1000, (runtime_ver % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", device_prop.major, device_prop.minor);
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
                 (unsigned long long)device_prop.totalGlobalMem);
        printf("%s", msg);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n", device_prop.multiProcessorCount,
               _ConvertSMVer2Cores(device_prop.major, device_prop.minor),
               _ConvertSMVer2Cores(device_prop.major, device_prop.minor) * device_prop.multiProcessorCount);
        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               clockRate * 1e-3f, clockRate * 1e-6f);
#if CUDART_VERSION >= 5000
        int memoryClockRate;
    #if CUDART_VERSION >= 13000
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev);
    #else
        memoryClockRate = device_prop.memoryClockRate;
    #endif
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", device_prop.memoryBusWidth);

        if (device_prop.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", device_prop.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               device_prop.maxTexture1D, device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
               device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1],
               device_prop.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %zu bytes\n", device_prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", device_prop.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", device_prop.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", device_prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", device_prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", device_prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", device_prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", device_prop.maxThreadsDim[0],
               device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", device_prop.maxGridSize[0],
               device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", device_prop.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", device_prop.textureAlignment);
        int gpuOverlap;
        cudaDeviceGetAttribute(&gpuOverlap, cudaDevAttrGpuOverlap, dev);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (gpuOverlap ? "Yes" : "No"), device_prop.asyncEngineCount);
        int kernelExecTimeout;
        cudaDeviceGetAttribute(&kernelExecTimeout, cudaDevAttrKernelExecTimeout, dev);
        printf("  Run time limit on kernels:                     %s\n", kernelExecTimeout ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", device_prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", device_prop.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", device_prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               device_prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n", device_prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n", device_prop.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               device_prop.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", device_prop.cooperativeLaunch ? "Yes" : "No");
        // The property cooperativeMultiDeviceLaunch is deprecated in CUDA 13.0
#if CUDART_VERSION < 13000
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
#endif
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", device_prop.pciDomainID,
               device_prop.pciBusID, device_prop.pciDeviceID);

        const char *sComputeMode[] = {"Default (multiple host threads can use ::cudaSetDevice() with device "
                                      "simultaneously)",
                                      "Exclusive (only one host thread in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Prohibited (no host thread can use ::cudaSetDevice() with this "
                                      "device)",
                                      "Exclusive Process (many threads in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Unknown",
                                      NULL};
        int computeMode;
        cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev);
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[computeMode]);
    }
}