#ifndef SKYWEAVER_BEAMFORMER_UTILS_CUH
#define SKYWEAVER_BEAMFORMER_UTILS_CUH

#include "cuComplex.h"
#include "skyweaver/types.cuh"

#include <type_traits>

namespace skyweaver
{

/**
 * @brief      Data structure for holding reshaped int8 complex data
 *             for use in the DP4A transform.
 *
 * @details    Typically this stores the data from one polarisation for
 *             4 antennas in r0,r1,r2,r3,i0,i1,i2,i3 order.
 */
struct char4x2 {
    char4 x;
    char4 y;
};

/**
 * @brief      Data structure for holding antenna voltages for transpose.
 *
 * @details    Typically this stores the data from one polarisation for
 *             4 antennas in r0,i0,r1,i1,r2,i2,r3,i3 order.
 */
struct char2x4 {
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
 * @note       The assembly instruction that underpins this operation
 * (dp4a.s32.s32).
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

enum StokesParameter { I, Q, U, V };

template <bool flag = false>
void static_no_match()
{
    static_assert(flag, "no match");
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
template <StokesParameter Stokes>
static inline __host__ __device__ float
calculate_stokes(cuFloatComplex const& p0, cuFloatComplex const& p1)
{
    if constexpr(Stokes == StokesParameter::I) {
        return cuCmagf(p0) + cuCmagf(p1);
    } else if constexpr(Stokes == StokesParameter::Q) {
        return cuCmagf(p0) - cuCmagf(p1);
    } else if constexpr(Stokes == StokesParameter::U) {
        return 2 * cuCrealf(cuCmulf(p0, cuConjf(p1)));
    } else if constexpr(Stokes == StokesParameter::V) {
        return 2 * cuCimagf(cuCmulf(p0, cuConjf(p1)));
    } else {
        static_no_match();
    }
}

template <StokesParameter Stokes>
struct SingleStokesBeamformerTraits {
    typedef int8_t QuantisedPowerType;
    typedef float RawPowerType;
    constexpr static const float zero_power = 0.0f;

    static inline __host__ __device__ void
    integrate_stokes(cuFloatComplex const& p0,
                     cuFloatComplex const& p1,
                     RawPowerType& power)
    {
        power += calculate_stokes<Stokes>(p0, p1);
    }

    static inline __host__ __device__ void
    integrate_weighted_stokes(cuFloatComplex const& p0,
                              cuFloatComplex const& p1,
                              RawPowerType& power,
                              float const& weight)
    {
        power += calculate_stokes<Stokes>(p0, p1) * weight;
    }

    static inline __host__ __device__ RawPowerType
    ib_subtract(RawPowerType const& power,
                RawPowerType const& ib_power,
                float const& ib_mutliplier,
                float const& scale_factor)
    {
        return rintf((power - ib_power * ib_mutliplier) / scale_factor);
    }

    static inline __host__ __device__ RawPowerType
    rescale(RawPowerType const& power,
            float const& offset,
            float const& scale_factor)
    {
        if constexpr(Stokes == StokesParameter::I) {
            return rintf((power - offset) / scale_factor);
        } else {
            return rintf(power / scale_factor);
        }
    }

    static inline __host__ __device__ QuantisedPowerType
    clamp(RawPowerType const& power)
    {
        return static_cast<QuantisedPowerType>(
            fmaxf(static_cast<float>(
                      std::numeric_limits<QuantisedPowerType>::lowest()),
                  fminf(static_cast<float>(
                            std::numeric_limits<QuantisedPowerType>::max()),
                        power)));
    }
};

struct FullStokesBeamformerTraits {
    typedef char4 QuantisedPowerType;
    typedef float4 RawPowerType;
    typedef StokesParameter Sp;
    constexpr static const float4 zero_power = {0.0f, 0.0f, 0.0f, 0.0f};
    template <Sp Stokes>
    using SSBfTraits = SingleStokesBeamformerTraits<Stokes>;

    static inline __host__ __device__ void
    integrate_stokes(cuFloatComplex const& p0,
                     cuFloatComplex const& p1,
                     RawPowerType& power)
    {
        SSBfTraits<Sp::I>::integrate_stokes(p0, p1, power.x);
        SSBfTraits<Sp::Q>::integrate_stokes(p0, p1, power.y);
        SSBfTraits<Sp::U>::integrate_stokes(p0, p1, power.z);
        SSBfTraits<Sp::V>::integrate_stokes(p0, p1, power.w);
    }

    static inline __host__ __device__ void
    integrate_weighted_stokes(cuFloatComplex const& p0,
                              cuFloatComplex const& p1,
                              RawPowerType& power,
                              float const& weight)
    {
        SSBfTraits<Sp::I>::integrate_weighted_stokes(p0, p1, power.x, weight);
        SSBfTraits<Sp::Q>::integrate_weighted_stokes(p0, p1, power.y, weight);
        SSBfTraits<Sp::U>::integrate_weighted_stokes(p0, p1, power.z, weight);
        SSBfTraits<Sp::V>::integrate_weighted_stokes(p0, p1, power.w, weight);
    }

    static inline __host__ __device__ RawPowerType
    ib_subtract(RawPowerType const& power,
                RawPowerType const& ib_power,
                float const& ib_mutliplier,
                float const& scale_factor)
    {
        RawPowerType subtracted;
        subtracted.x = SSBfTraits<Sp::I>::ib_subtract(power.x,
                                                      ib_power.x,
                                                      ib_mutliplier,
                                                      scale_factor);
        subtracted.y = SSBfTraits<Sp::Q>::ib_subtract(power.y,
                                                      ib_power.y,
                                                      ib_mutliplier,
                                                      scale_factor);
        subtracted.z = SSBfTraits<Sp::U>::ib_subtract(power.z,
                                                      ib_power.z,
                                                      ib_mutliplier,
                                                      scale_factor);
        subtracted.w = SSBfTraits<Sp::V>::ib_subtract(power.w,
                                                      ib_power.w,
                                                      ib_mutliplier,
                                                      scale_factor);
        return subtracted;
    }

    static inline __host__ __device__ RawPowerType
    rescale(RawPowerType const& power,
            float const& offset,
            float const& scale_factor)
    {
        RawPowerType rescaled;
        rescaled.x = SSBfTraits<Sp::I>::rescale(power.x, offset, scale_factor);
        rescaled.y = SSBfTraits<Sp::Q>::rescale(power.y, offset, scale_factor);
        rescaled.z = SSBfTraits<Sp::U>::rescale(power.z, offset, scale_factor);
        rescaled.w = SSBfTraits<Sp::V>::rescale(power.w, offset, scale_factor);
        return rescaled;
    }

    static inline __host__ __device__ QuantisedPowerType
    clamp(RawPowerType const& power)
    {
        QuantisedPowerType clamped;
        clamped.x = SSBfTraits<Sp::I>::clamp(power.x);
        clamped.y = SSBfTraits<Sp::Q>::clamp(power.y);
        clamped.z = SSBfTraits<Sp::U>::clamp(power.z);
        clamped.w = SSBfTraits<Sp::V>::clamp(power.w);
        return clamped;
    }
};

} // namespace skyweaver

#endif // SKYWEAVER_BEAMFORMER_UTILS_CUH