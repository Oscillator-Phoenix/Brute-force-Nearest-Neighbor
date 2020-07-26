#include "core.h"
#include <cstdio>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


extern void cudaCallbackCPU(int k, int m, int n, float *searchPoints,
                            float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k*mInd+kInd] - referencePoints[k*nInd+kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        tmp[mInd] = minIndex;
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}

// Just a demo. Actually the same to the CPU version.
extern void cudaCallbackGPU(int k, int m, int n, float *searchPoints,
                            float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k*mInd+kInd] - referencePoints[k*nInd+kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        tmp[mInd] = minIndex;
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}



__device__ int device_divup(int n, int m) {
    return n % m == 0 ? n/m : n/m + 1;
}

void naive_cpp_version(int k, int m, int n, float *searchPoints,
                              float *referencePoints, int *results) 
{
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k*mInd+kInd] - referencePoints[k*nInd+kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        results[mInd] = minIndex;
    }
}


// 核函数：查询1点的基础版本
// 返回 tmp_dis，是第 mInd 个 searchPoint 到 n 个 referencePoint 的距离
__global__ void k_naive_1_nn(const int k, 
                             const int mInd, 
                             const int n,
                             const float *searchPoints,
                             const float *referencePoints,
                             float *tmp_dis) 
{    
    int nInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (nInd >= n) return;
    
    int searchPointStart    = mInd*k;
    int referencePointStart = nInd*k;
    float squareSum = 0.0;
    
    for(int kInd = 0; kInd < k; kInd++) {
        float diff = searchPoints[searchPointStart+kInd] - referencePoints[referencePointStart+kInd];
        squareSum += (diff*diff);
    }
    
    tmp_dis[nInd] = squareSum;
}


// 查询1个点的基础版本
void navie_1_nn(int k, int m, int n, float *searchPoints,
                float *referencePoints, int *results) 
{
    float *d_searchPoints;
    float *d_referencePoints;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&d_searchPoints,    sizeof(float) * k * m));
    CHECK(cudaMalloc((void **)&d_referencePoints, sizeof(float) * k * n));
    
    CHECK(cudaMemcpy(d_searchPoints,    searchPoints,    sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_referencePoints, referencePoints, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    float *d_tmp_dis;
    CHECK(cudaMalloc((void **)&d_tmp_dis, sizeof(float) * n));
    
    const int TPB_X = 32;
    dim3 block( TPB_X,           1, 1 );
    dim3 grid ( divup(n, TPB_X), 1, 1);   //  n / TPB_X
    // printf("block(%d, %d, %d), grid(%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    
    for(int mInd = 0; mInd < m; mInd++) {
        k_naive_1_nn<<< grid, block >>>(k, mInd, n, d_searchPoints, d_referencePoints, d_tmp_dis);
        cudaDeviceSynchronize();
        
        float *min = thrust::min_element(thrust::device, d_tmp_dis, d_tmp_dis + n);
        results[mInd] = min - d_tmp_dis;
    }
    
    CHECK(cudaFree(d_searchPoints));
    CHECK(cudaFree(d_referencePoints));
    CHECK(cudaFree(d_tmp_dis));
}             



// 核函数：查询1024个点的基础版本
__global__ void k_navie_1024_nn(const int k, 
                                const int m, 
                                const int n,
                                const float *searchPoints,
                                const float *referencePoints,
                                int *results)
{
    int mInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (mInd >= m) return;
    
    int searchPointStart = mInd*k;
    float minSquareSum = -1.0;
    int minIndex;
    
    for(int nInd = 0; nInd < n; nInd++) {
        int referencePointStart = nInd*k;
        float squareSum = 0.0;
        for(int kInd = 0; kInd < k; kInd++) {
            float diff = searchPoints[searchPointStart+kInd] - referencePoints[referencePointStart+kInd];
            squareSum += (diff*diff);
        }
        if (minSquareSum == -1.0 || squareSum < minSquareSum) {
            minSquareSum = squareSum;
            minIndex = nInd;
        }
    }
    
    results[mInd] = minIndex;
}


// 查询1024个点的基础版本
void navie_1024_nn(int k, int m, int n, float *searchPoints,
                        float *referencePoints, int *results) 
{
    float *d_searchPoints;
    float *d_referencePoints;
    int   *d_results;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&d_searchPoints,    sizeof(float) * k * m));
    CHECK(cudaMalloc((void **)&d_referencePoints, sizeof(float) * k * n));
    CHECK(cudaMalloc((void **)&d_results,         sizeof(int) * m));
    
    CHECK(cudaMemcpy(d_searchPoints,    searchPoints,    sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_referencePoints, referencePoints, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    const int TPB_X = 256;
    
    dim3 block( TPB_X,           1, 1 );
    dim3 grid ( divup(m, TPB_X), 1, 1);
    // printf("block(%d, %d, %d), grid(%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    
    k_navie_1024_nn<<< grid, block >>>(k, m, n, d_searchPoints, d_referencePoints, d_results);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    CHECK(cudaMemcpy(results, d_results, sizeof(int) * m, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_searchPoints));
    CHECK(cudaFree(d_referencePoints));
    CHECK(cudaFree(d_results));
}                           



// 核函数：共享内存 + 查询1024个点的基础版本（有最高维度限制）
#define MAX_DIM 64
#define MAX_SMEM_FLOAT_SZ (((48*1024)/sizeof(float)))

__global__ void k_smem_navie_1024_nn(const int k, 
                                     const int m, 
                                     const int n,
                                     const float *  __restrict__ searchPoints,
                                     const float *  __restrict__ referencePoints,
                                     int *  __restrict__  results)
{
    int mInd = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load my SearchPoint
    float mySearchPoint[MAX_DIM];
    if (mInd < m) {
        #pragma unroll 8
        for(int kInd = 0; kInd < k; kInd++) {
            mySearchPoint[kInd] = searchPoints[mInd*k + kInd];
        }
    }
    
    float minSquareSum = -1.0;
    int minIndex;
    
    // shared mem to load ReferencePoints
    __shared__ float smem[MAX_SMEM_FLOAT_SZ];
    
    int numReferencePointPerBatch = MAX_SMEM_FLOAT_SZ / k;
    int batchCount                = device_divup(n, numReferencePointPerBatch); 
    
    // if (mInd == 0) printf("numReferencePointPerBatch %d, batchCount %d\n", numReferencePointPerBatch, batchCount);

    for(int batchId = 0; batchId < batchCount; batchId++)
    {
        int batchStart = batchId * numReferencePointPerBatch;
        int batchLen   = (batchStart+numReferencePointPerBatch) < n ? numReferencePointPerBatch : (n-batchStart);
        
        // load gmem to smem with all threads in the block
        __syncthreads();
        int numFloat        = batchLen   * k;
        int floatBatchStart = batchStart * k;
        #pragma unroll 8
        for(int i = threadIdx.x; i < numFloat; i += blockDim.x) {
            smem[i] = referencePoints[floatBatchStart + i];
        }
        __syncthreads();
        
        // if (k == 3 && m <= 2 && n <= 8) {
        //     if (mInd == 0) {
        //         printf("numFloat %d\n", numFloat);
        //         printf("blockDim.x %d\n", blockDim.x);
        //         printf("floatBatchStart %d\n", floatBatchStart);
        //         for(int i = 0; i < numFloat; i++) {
        //             float x = smem[i];
        //             printf("%f ", x);
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();
        
        // calculate nn only with valid threads
        if (mInd < m) {
            for(int nInd = 0; nInd < batchLen; nInd++) {
                int smemStart = nInd * k;
                float squareSum = 0.0;
                
                #pragma unroll 8
                for(int kInd = 0; kInd < k; kInd++) {
                    float diff = mySearchPoint[kInd] - smem[smemStart + kInd];
                    // float diff = mySearchPoint[kInd] - referencePoints[(batchStart+nInd)*k+kInd];
                    squareSum += (diff*diff);
                }
                if (minSquareSum == -1.0 || squareSum < minSquareSum) {
                    minSquareSum = squareSum;
                    minIndex = batchStart + nInd;
                }
            }
        }

    }

    if (mInd < m) {
        results[mInd] = minIndex;
    }
}



// 共享内存优化 + 查询1024个点的基础版本
void smem_navie_1024_nn(int k, int m, int n, float *searchPoints,
                        float *referencePoints, int *results) 
{
    float *d_searchPoints;
    float *d_referencePoints;
    int   *d_results;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&d_searchPoints,    sizeof(float) * k * m));
    CHECK(cudaMalloc((void **)&d_referencePoints, sizeof(float) * k * n));
    CHECK(cudaMalloc((void **)&d_results,         sizeof(int) * m));
    
    CHECK(cudaMemcpy(d_searchPoints,    searchPoints,    sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_referencePoints, referencePoints, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    const int TPB_X = 32;
    
    dim3 block( TPB_X,           1, 1 );
    dim3 grid ( divup(m, TPB_X), 1, 1);
    // printf("block(%d, %d, %d), grid(%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    
    k_smem_navie_1024_nn<<< grid, block >>>(k, m, n, d_searchPoints, d_referencePoints, d_results);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    CHECK(cudaMemcpy(results, d_results, sizeof(int) * m, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_searchPoints));
    CHECK(cudaFree(d_referencePoints));
    CHECK(cudaFree(d_results));
}             
                         


// 核函数：常量内存 + 查询1024个点的基础版本
#define MAX_CMEM_FLOAT_SZ ((64*1024)/sizeof(double))

__constant__ float dc_referencePoints[MAX_CMEM_FLOAT_SZ];

__global__ void k_cmem_navie_1024_nn(const int k, 
                                     const int m, 
                                     const int n,
                                     const int batchStart,
                                     const int batchLen,
                                     const float *searchPoints,
                                     float *tmpMinDis,
                                     int *results)
{
    int mInd = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mInd < m) {
        
        // load my SearchPoint
        float mySearchPoint[MAX_DIM];
        #pragma unroll 8
        for(int kInd = 0; kInd < k; kInd++) {
            mySearchPoint[kInd] = searchPoints[mInd*k + kInd];
        }
        
        float minSquareSum = -1.0;
        int minIndex;
    
        
        for(int nInd = 0; nInd < batchLen; nInd++) {
        
            int   cmemStart = nInd*k;
            float squareSum = 0.0;
        
            #pragma unroll 8
            for(int kInd = 0; kInd < k; kInd++) {
                float diff = mySearchPoint[kInd] - dc_referencePoints[cmemStart + kInd];
                squareSum += (diff*diff);
            }
        
            if (minSquareSum == -1.0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = batchStart + nInd;
            }
        }
    
        if (tmpMinDis[mInd] == -1.0 || minSquareSum < tmpMinDis[mInd]) {
            tmpMinDis[mInd] = minSquareSum;
            results[mInd] = minIndex;
        }
    }
}

// 常量内存 + 查询1024个点的基础版本查询1024个点的基础版本
void cmem_navie_1024_nn(int k, int m, int n, float *searchPoints,
                        float *referencePoints, int *results) 
{
    float *d_searchPoints;
    int   *d_results;
    float *d_tmpMinDis;
    
    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&d_searchPoints,    sizeof(float) * k * m));
    CHECK(cudaMalloc((void **)&d_results,         sizeof(int) * m));
    CHECK(cudaMalloc((void **)&d_tmpMinDis,       sizeof(float) * m));
    
    CHECK(cudaMemcpy(d_searchPoints, searchPoints, sizeof(float) * k * m, cudaMemcpyHostToDevice));
    thrust::fill_n(thrust::device, d_tmpMinDis, m, -1.0);
    
    const int TPB_X = 32;
    dim3 block( TPB_X,           1, 1 );
    dim3 grid ( divup(m, TPB_X), 1, 1);
    
    int numReferencePointPerBatch = MAX_CMEM_FLOAT_SZ / k;
    int batchCount                = divup(n, numReferencePointPerBatch); 
    
    for(int batchId = 0; batchId < batchCount; batchId++)
    {
        int batchStart = batchId * numReferencePointPerBatch;
        int batchLen   = (batchStart+numReferencePointPerBatch) < n ? numReferencePointPerBatch : (n-batchStart);
        
        // printf("numReferencePointPerBatch %d, batchStart %d, batchLen %d, k %d, nbyte %d \n", numReferencePointPerBatch, batchStart, batchLen, k, batchLen * k * (sizeof(float)));
        
        // load host data to device cmem
        CHECK(cudaMemcpyToSymbol(dc_referencePoints, 
                                 referencePoints + (batchStart*k), 
                                 batchLen * k * (sizeof(float)), 
                                 0, cudaMemcpyHostToDevice));
        
        k_cmem_navie_1024_nn<<< grid, block >>>(k, m, n, batchStart, batchLen, d_searchPoints, d_tmpMinDis, d_results);
    }
    cudaDeviceSynchronize();
    

    // Copy back the results and de-allocate the device memory
    CHECK(cudaMemcpy(results, d_results, sizeof(int) * m, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_searchPoints));
    CHECK(cudaFree(d_results));
    CHECK(cudaFree(d_tmpMinDis));
}                           




extern void cudaCallback(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);

    // naive_cpp_version  (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1_nn         (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1024_nn      (k, m, n, searchPoints, referencePoints, tmp);
    // smem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);
    cmem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
} 


extern void cudaCallbackNaive1NN(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);

    // naive_cpp_version  (k, m, n, searchPoints, referencePoints, tmp);
    navie_1_nn         (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1024_nn      (k, m, n, searchPoints, referencePoints, tmp);
    // smem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);
    // cmem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}

extern void cudaCallbackNaive1024NN(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);

    // naive_cpp_version  (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1_nn         (k, m, n, searchPoints, referencePoints, tmp);
    navie_1024_nn      (k, m, n, searchPoints, referencePoints, tmp);
    // smem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);
    // cmem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}


extern void cudaCallbackSmemNaive1024NN(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);

    // naive_cpp_version  (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1_nn         (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1024_nn      (k, m, n, searchPoints, referencePoints, tmp);
    smem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);
    // cmem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}


extern void cudaCallbackCmemNaive1024NN(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);

    // naive_cpp_version  (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1_nn         (k, m, n, searchPoints, referencePoints, tmp);
    // navie_1024_nn      (k, m, n, searchPoints, referencePoints, tmp);
    // smem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);
    cmem_navie_1024_nn (k, m, n, searchPoints, referencePoints, tmp);

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}












