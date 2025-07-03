#include <cuda_runtime.h>

inline __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float cross(float2 a, float2 b) {
  return a.x * b.y - a.y * b.x;
}

__device__ bool intersection1d(float2 a, float2 b) {
  float amin = fminf(a.x, a.y);
  float amax = fmaxf(a.x, a.y);
  float bmin = fminf(b.x, b.y);
  float bmax = fmaxf(b.x, b.y);
  return fmaxf(amin, bmin) <= fminf(amax, bmax);
}

__device__ bool intersection2d(float2 a1, float2 a2, float2 b1, float2 b2) {
  if (!intersection1d(make_float2(a1.x, a2.x), make_float2(b1.x, b2.x)) ||
      !intersection1d(make_float2(a1.y, a2.y), make_float2(b1.y, b2.y))) {
    return false;
  }

  float2 a = a2 - a1;
  float2 b = b2 - b1;
  if (cross(a, b1 - a1) * cross(a, b2 - a1) <= 0 &&
      cross(b, a1 - b1) * cross(b, a2 - b1) <= 0) {
    return true;
  }

  return false;
}

__device__ bool intersection2d(float2 a1, float2 a2, float2 b1, float2 b2, float2 &x) {
  if (!intersection1d(make_float2(a1.x, a2.x), make_float2(b1.x, b2.x)) ||
      !intersection1d(make_float2(a1.y, a2.y), make_float2(b1.y, b2.y))) {
    return false;
  }

  float2 a = a2 - a1;
  float2 b = b2 - b1;
  float2 c = b1 - a1;
  float denom = cross(a, b);
  if (denom == 0.f) return false;  // parallel or overlap

  float s = cross(c, b) / denom;
  float t = cross(c, a) / denom;
  if (t >= 0.f && t <= 1.f && s >= 0.f && s <= 1.f) {
    x = a1 + a * s;
    return true;
  }

  return false;
}