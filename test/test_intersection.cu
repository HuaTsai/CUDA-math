#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <vector>

// Include the functions from intersection.cu
#include "../src/intersection.cu"

void check(cudaError_t result, const char* file, int line) {
  if (result != cudaSuccess) {
    FAIL() << "CUDA error at " << file << ":" << line << " - "
           << cudaGetErrorString(result) << " (code " << result << ")";
  }
}

#define checkCudaErrors(val) check((val), __FILE__, __LINE__)

// CUDA kernel function: test cross function
__global__ void test_cross_kernel(float2* a, float2* b, float* result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = cross(a[idx], b[idx]);
  }
}

// CUDA kernel function: test intersection1d function
__global__ void test_intersection1d_kernel(float2* a, float2* b, bool* result,
                                           int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = intersection1d(a[idx], b[idx]);
  }
}

// CUDA kernel function: test intersection2d function (without returning
// intersection point)
__global__ void test_intersection2d_kernel(float2* a1, float2* a2, float2* b1,
                                           float2* b2, bool* result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = intersection2d(a1[idx], a2[idx], b1[idx], b2[idx]);
  }
}

// CUDA kernel function: test intersection2d function (with returning
// intersection point)
__global__ void test_intersection2d_with_point_kernel(
    float2* a1, float2* a2, float2* b1, float2* b2, bool* result,
    float2* intersection_points, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float2 x;
    result[idx] = intersection2d(a1[idx], a2[idx], b1[idx], b2[idx], x);
    intersection_points[idx] = x;
  }
}

// Google Test class
class CUDAMathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    ASSERT_GT(deviceCount, 0) << "No CUDA device found!";

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
  }

  void TearDown() override {
    // Clean up CUDA resources
    cudaDeviceReset();
  }
};

// Test cross function
TEST_F(CUDAMathTest, CrossFunction) {
  const int n = 5;
  std::vector<float2> a_host = {
      {1.0f, 0.0f}, {3.0f, 4.0f}, {1.0f, 1.0f}, {-2.0f, 3.0f}, {0.0f, 1.0f}};
  std::vector<float2> b_host = {
      {0.0f, 1.0f}, {1.0f, 2.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
  std::vector<float> expected = {1.0f, 2.0f, -2.0f, -5.0f, -1.0f};

  float2 *a_dev, *b_dev;
  float* result_dev;
  std::vector<float> result_host(n);

  checkCudaErrors(cudaMalloc(&a_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&result_dev, n * sizeof(float)));

  checkCudaErrors(cudaMemcpy(a_dev, a_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b_dev, b_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));

  test_cross_kernel<<<1, n>>>(a_dev, b_dev, result_dev, n);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(result_host.data(), result_dev, n * sizeof(float),
                             cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(result_host[i], expected[i], 1e-6)
        << "cross({" << a_host[i].x << ", " << a_host[i].y << "}, {"
        << b_host[i].x << ", " << b_host[i].y << "}) test failed";
  }

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(result_dev);
}

// Test intersection1d function
TEST_F(CUDAMathTest, Intersection1DFunction) {
  const int n = 6;
  std::vector<float2> a_host = {{1.0f, 3.0f}, {2.0f, 5.0f}, {1.0f, 2.0f},
                                {4.0f, 6.0f}, {0.0f, 1.0f}, {3.0f, 1.0f}};
  std::vector<float2> b_host = {{2.0f, 4.0f}, {1.0f, 3.0f}, {3.0f, 5.0f},
                                {7.0f, 8.0f}, {0.5f, 1.5f}, {2.0f, 4.0f}};
  std::vector<bool> expected = {true, true, false, false, true, true};

  float2 *a_dev, *b_dev;
  bool* result_dev;
  std::vector<bool> result_host(n);
  bool* result_host_ptr = new bool[n];

  checkCudaErrors(cudaMalloc(&a_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&result_dev, n * sizeof(bool)));

  checkCudaErrors(cudaMemcpy(a_dev, a_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b_dev, b_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));

  test_intersection1d_kernel<<<1, n>>>(a_dev, b_dev, result_dev, n);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(result_host_ptr, result_dev, n * sizeof(bool),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    result_host[i] = result_host_ptr[i];
  }

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(result_host[i], expected[i])
        << "intersection1d([" << a_host[i].x << ", " << a_host[i].y << "], ["
        << b_host[i].x << ", " << b_host[i].y << "]) test failed";
  }

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(result_dev);
  delete[] result_host_ptr;
}

// Test intersection2d function (without returning intersection point)
TEST_F(CUDAMathTest, Intersection2DFunction) {
  const int n = 5;
  std::vector<float2> a1_host = {
      {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 2.0f}, {-1.0f, -1.0f}};
  std::vector<float2> a2_host = {
      {2.0f, 2.0f}, {1.0f, 1.0f}, {3.0f, 3.0f}, {2.0f, 0.0f}, {1.0f, 1.0f}};
  std::vector<float2> b1_host = {
      {0.0f, 2.0f}, {2.0f, 0.0f}, {0.0f, 3.0f}, {3.0f, 1.0f}, {3.0f, 0.0f}};
  std::vector<float2> b2_host = {
      {2.0f, 0.0f}, {0.0f, 2.0f}, {3.0f, 0.0f}, {1.0f, 3.0f}, {4.0f, 1.0f}};
  std::vector<bool> expected = {true, true, true, false, false};

  float2 *a1_dev, *a2_dev, *b1_dev, *b2_dev;
  bool* result_dev;
  std::vector<bool> result_host(n);
  bool* result_host_ptr = new bool[n];

  checkCudaErrors(cudaMalloc(&a1_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&a2_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b1_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b2_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&result_dev, n * sizeof(bool)));

  checkCudaErrors(cudaMemcpy(a1_dev, a1_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(a2_dev, a2_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b1_dev, b1_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b2_dev, b2_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));

  test_intersection2d_kernel<<<1, n>>>(a1_dev, a2_dev, b1_dev, b2_dev,
                                       result_dev, n);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(result_host_ptr, result_dev, n * sizeof(bool),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    result_host[i] = result_host_ptr[i];
  }

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(result_host[i], expected[i])
        << "intersection2d((" << a1_host[i].x << "," << a1_host[i].y << ")-("
        << a2_host[i].x << "," << a2_host[i].y << "), (" << b1_host[i].x << ","
        << b1_host[i].y << ")-(" << b2_host[i].x << "," << b2_host[i].y
        << ")) test failed";
  }

  cudaFree(a1_dev);
  cudaFree(a2_dev);
  cudaFree(b1_dev);
  cudaFree(b2_dev);
  cudaFree(result_dev);
  delete[] result_host_ptr;
}

// Test intersection2d function (with returning intersection point)
TEST_F(CUDAMathTest, Intersection2DWithPointFunction) {
  const int n = 3;
  std::vector<float2> a1_host = {{0.0f, 0.0f}, {-1.0f, 0.0f}, {1.0f, 1.0f}};
  std::vector<float2> a2_host = {{2.0f, 2.0f}, {1.0f, 0.0f}, {3.0f, 3.0f}};
  std::vector<float2> b1_host = {{0.0f, 2.0f}, {0.0f, -1.0f}, {0.0f, 3.0f}};
  std::vector<float2> b2_host = {{2.0f, 0.0f}, {0.0f, 1.0f}, {3.0f, 0.0f}};
  std::vector<bool> expected = {true, true, true};
  std::vector<float2> expected_points = {
      {1.0f, 1.0f}, {0.0f, 0.0f}, {1.5f, 1.5f}};

  float2 *a1_dev, *a2_dev, *b1_dev, *b2_dev, *points_dev;
  bool* result_dev;
  std::vector<bool> result_host(n);
  std::vector<float2> points_host(n);
  bool* result_host_ptr = new bool[n];

  checkCudaErrors(cudaMalloc(&a1_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&a2_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b1_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&b2_dev, n * sizeof(float2)));
  checkCudaErrors(cudaMalloc(&result_dev, n * sizeof(bool)));
  checkCudaErrors(cudaMalloc(&points_dev, n * sizeof(float2)));

  checkCudaErrors(cudaMemcpy(a1_dev, a1_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(a2_dev, a2_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b1_dev, b1_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b2_dev, b2_host.data(), n * sizeof(float2),
                             cudaMemcpyHostToDevice));

  test_intersection2d_with_point_kernel<<<1, n>>>(
      a1_dev, a2_dev, b1_dev, b2_dev, result_dev, points_dev, n);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(result_host_ptr, result_dev, n * sizeof(bool),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(points_host.data(), points_dev, n * sizeof(float2),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    result_host[i] = result_host_ptr[i];
  }

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(result_host[i], expected[i])
        << "Test " << i + 1
        << " intersection2d returned wrong intersection result";
    if (result_host[i]) {
      EXPECT_NEAR(points_host[i].x, expected_points[i].x, 1e-6)
          << "Test " << i + 1
          << " intersection point x-coordinate is incorrect";
      EXPECT_NEAR(points_host[i].y, expected_points[i].y, 1e-6)
          << "Test " << i + 1
          << " intersection point y-coordinate is incorrect";
    }
  }

  cudaFree(a1_dev);
  cudaFree(a2_dev);
  cudaFree(b1_dev);
  cudaFree(b2_dev);
  cudaFree(result_dev);
  cudaFree(points_dev);
  delete[] result_host_ptr;
}
