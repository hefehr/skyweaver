#ifndef SKYWEAVER_BEAMFORMER_UTILS_CUH
#define SKYWEAVER_BEAMFORMER_UTILS_CUH

#include "cuComplex.h"

/**
 * @brief      Data structure for holding reshaped int8 complex data
 *             for use in the DP4A transform. 
 * 
 * @details    Typically this stores the data from one polarisation for 
 *             4 antennas in r0,r1,r2,r3,i0,i1,i2,i3 order.
 */
struct char4x2
{
    char4 x;
    char4 y;
};


/**
 * @brief      Data structure for holding antenna voltages for transpose.
 * 
 * @details    Typically this stores the data from one polarisation for 
 *             4 antennas in r0,i0,r1,i1,r2,i2,r3,i3 order.
 */
struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

/**
 * @brief      Wrapper for the DP4A int8 fused multiply add instruction
 *
 * @param      c     The output value
 * @param[in]  a     An integer composed of 4 chars
 * @param[in]  b     An integer composed of 4 chars
 *
 * @detail     If we treat a and b like to char4 instances, then the dp4a
 *             instruction performs the following:
 *
 *             c = (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w)
 *
 * @note       The assembly instruction that underpins this operation (dp4a.s32.s32).
 *             
 */
__forceinline__ __device__ void dp4a(int& c, const int& a, const int& b)
{
#if __CUDA_ARCH__ >= 610
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
    char4& a4 = *((char4*)&a);
    char4& b4 = *((char4*)&b);
    c += a4.x * b4.x;
    c += a4.y * b4.y;
    c += a4.z * b4.z;
    c += a4.w * b4.w;
#endif
}

/**
 * @brief      Transpose an int2 from a char2x4 to a char4x2.
 *
 * @param      input  The value to transpose
 *
 * @note       This is used to go from (for 4 sequential antennas):
 *
 *             [[real, imag],
 *              [real, imag],
 *              [real, imag],
 *              [real, imag]]
 *
 *             to
 *
 *             [[real, real, real, real],
 *              [imag, imag, imag, imag]]
 */
__forceinline__ __device__ int2 int2_transpose(int2 const& input)
{
    char2x4 a;
    char4x2 b;
    a     = (*(char2x4*)&input);
    b.x.x = a.x.x;
    b.x.y = a.y.x;
    b.x.z = a.z.x;
    b.x.w = a.w.x;
    b.y.x = a.x.y;
    b.y.y = a.y.y;
    b.y.z = a.z.y;
    b.y.w = a.w.y;
    return (*(int2*)&b);
}

/**
 * @brief Calculate the square of a complex number
 */
__host__ __device__ static __inline__ float cuCmagf(cuFloatComplex x)
{
    return x.x * x.x + x.y * x.y;
}

/**
 * @brief Calculate the define Stokes parameter
 * 
 * Stokes modes can be considered here:
 * I = P0^2 + P1^2
 * Q = P0^2 - P1^2
 * U = 2 * Re(P0 * conj(P1))
 * V = 2 * Im(P0 * conj(P1))
 */
__host__ __device__ static __inline__ float calculate_stokes(cuFloatComplex p0,
                                                             cuFloatComplex p1)
{
#if SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_I
    return cuCmagf(p0) + cuCmagf(p1);
#elif SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_Q
    return cuCmagf(p0) - cuCmagf(p1);
#elif SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_U
    return 2 * cuCrealf(cuCmulf(p0, cuConjf(p1)));
#elif SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_V
    return 2 * cuCimagf(cuCmulf(p0, cuConjf(p1)));
#else
    static_assert(false,
                  "Invalid Stokes mode defined. Must be one "
                  "of 0 (I), 1 (Q), 2 (U) or 3 (V)");
#endif
}

#endif //SKYWEAVER_BEAMFORMER_UTILS_CUH