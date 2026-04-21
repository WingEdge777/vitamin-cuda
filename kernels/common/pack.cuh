#pragma once
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>

#define UINT2(value) (reinterpret_cast<uint2 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

const int warp_size = 32;

union alignas(16) pack128 {
    float4 f4;
    float f[4];
    int4 i4;
    int8_t i8[16];
    half2 h2[4];
    half h[8];
    __nv_bfloat162 bf2[4];
    __nv_bfloat16 bf[8];
};

union alignas(8) pack64 {
    float2 f2;
    float f[2];
    int2 i4;
    __nv_bfloat162 bf2[2];
    __nv_bfloat16 bf[4];
};
