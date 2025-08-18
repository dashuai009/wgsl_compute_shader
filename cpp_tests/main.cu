#include <iostream>

#include "svd3_cuda.h"

__device__ void svd3x3_device(const float *F, float *U, float *S, float *V) {
  // 从F数组中提取3x3矩阵的9个元素（按行存储）
  float a11 = F[0], a12 = F[1], a13 = F[2];
  float a21 = F[3], a22 = F[4], a23 = F[5];
  float a31 = F[6], a32 = F[7], a33 = F[8];

  // 定义输出变量（U、S、V的元素）
  float u11, u12, u13, u21, u22, u23, u31, u32, u33;
  float s11, s22, s33; // 奇异值（对角元素）
  float v11, v12, v13, v21, v22, v23, v31, v32, v33;

  // 调用第一份代码中的SVD接口
  svd(a11, a12, a13, a21, a22, a23, a31, a32, a33, u11, u12, u13, u21, u22, u23,
      u31, u32, u33, s11, s22, s33, v11, v12, v13, v21, v22, v23, v31, v32,
      v33);

  // 将结果整理到输出数组（按行存储）
  // U矩阵
  U[0] = u11;
  U[1] = u12;
  U[2] = u13;
  U[3] = u21;
  U[4] = u22;
  U[5] = u23;
  U[6] = u31;
  U[7] = u32;
  U[8] = u33;

  // 奇异值（对角元素）
  S[0] = s11;
  S[1] = s22;
  S[2] = s33;

  // V矩阵
  V[0] = v11;
  V[1] = v12;
  V[2] = v13;
  V[3] = v21;
  V[4] = v22;
  V[5] = v23;
  V[6] = v31;
  V[7] = v32;
  V[8] = v33;
}
__global__ void svd3x3_kernel(const float *F, float *U, float *S, float *V) {
  svd3x3_device(F, U, S, V);
}

void print_matrix(float *M) {
  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << M[i * 3 + j] << " ";
    }
    std::cout << std::endl;
  }
}
int main() {
  // float h_F[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float h_F[9] = {-3.9313042, -3.091185, 1.1816511, -1.3155389, 4.4590015, 4.946289, 4.4633255, 1.9406939, -0.75030804};
  float h_U[9], h_S[3], h_V[9];

  float *d_F, *d_U, *d_S, *d_V;
  cudaMalloc(&d_F, 9 * sizeof(float));
  cudaMalloc(&d_U, 9 * sizeof(float));
  cudaMalloc(&d_S, 3 * sizeof(float));
  cudaMalloc(&d_V, 9 * sizeof(float));

  cudaMemcpy(d_F, h_F, 9 * sizeof(float), cudaMemcpyHostToDevice);

  svd3x3_kernel<<<1, 1>>>(d_F, d_U, d_S, d_V);
  cudaDeviceSynchronize();

  cudaMemcpy(h_U, d_U, 9 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_S, d_S, 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_V, d_V, 9 * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "U:\n";
  print_matrix(h_U);
  std::cout << "S: " << h_S[0] << " " << h_S[1] << " " << h_S[2] << "\n";
  std::cout << "V:\n";
  print_matrix(h_V);

  cudaFree(d_F);
  cudaFree(d_U);
  cudaFree(d_S);
  cudaFree(d_V);
}