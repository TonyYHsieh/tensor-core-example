#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        cudaError_t error = cmd;                                                                    \
        if (error != cudaSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

inline __device__ void ldsm(uint32_t &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(dst) : "r"(ptr));
#endif
}

inline __device__ void ldsmt(uint32_t &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(dst) : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint2 &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#endif
}

inline __device__ void ldsmt(uint2 &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst.x), "=r"(dst.y) : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsm(uint4 &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#endif
}

inline __device__ void ldsmt(uint4 &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#endif
}

#define N 32

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void ldsm_t_kernel(T *out, T *in){
    __shared__ T data[128];

    auto tid { threadIdx.x };

    if (tid < N) {
        data[tid] = in[tid];

        T tmp;
        ldsm(tmp, __nvvm_get_smem_pointer(&data[tid*4]));

        //printf("tid: %d (smem addr: %d) => in: %d, data: %d, tmp = %d\n", tid, smem_addr, in[tid], data[tid], tmp);
        out[tid] = tmp;
    }
}

int main() {
    using namespace std;
    using T = uint32_t;

    static int device = 0;
    CHECK(cudaSetDevice(device));
    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
    printf("info: compute compatibility of CUDA GPU device is: %d.%d\n", props.major, props.minor);

    T in_h[N];

    for (uint32_t idx = 0; idx < N; ++idx)
        in_h[idx] = idx;

    T out_h[N];

    T *in;
    T *out;
    CHECK(cudaMalloc(&in, sizeof(T) * N));
    CHECK(cudaMalloc(&out, sizeof(T) * N));
    CHECK(cudaMemcpy(in, in_h, sizeof(T) * N, cudaMemcpyDefault));

    const uint32_t threadsPerBlock {32};
    const uint32_t numBlocks {1};
    ldsm_t_kernel<<<numBlocks, threadsPerBlock>>>(out, in);

    CHECK(cudaMemcpy(out_h, out, sizeof(T) * N, cudaMemcpyDefault));

    for (auto idx = 0; idx < N; ++idx)
        cout << idx << ": " << out_h[idx] << dec << endl;

    CHECK(cudaFree(in));
    CHECK(cudaFree(out));

    return 0;
}
