#ifndef SKYWEAVER_BEAMFORMER_UTILS_CUH
#define SKYWEAVER_BEAMFORMER_UTILS_CUH

namespace {
    #define AT(var, idx) accessor<decltype(var)>::template at<idx>(var)
}

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


/**
 * Helpers for getting the underlying storage types
 * for sets of different Stokes parameters up from
 * 1 to 4.
 */
template <int N> struct stokes_storage_type {};
template <> struct stokes_storage_type<1> {
	using QuantisedPowerType = int8_t;
	using RawPowerType = float;
};
template <> struct stokes_storage_type<2> {
	using QuantisedPowerType = char2;
	using RawPowerType = float2;
};
template <> struct stokes_storage_type<3> {
	using QuantisedPowerType = char3;
	using RawPowerType = float3;
};
template <> struct stokes_storage_type<4> {
	using QuantisedPowerType = char4;
	using RawPowerType = float4;
};

/**
 * Helpers for getting the Nth value of a vector type by reference
 * An index of 0 turns into type::x, 1 --> type::y etc.
 * Direct usage of this struct should be avoided an instead the 
 * AT preprocessor macro should be used where AT(var, index) will
 * return a reference to the Nth index of var.   
 */
template <typename T>
struct accessor {
	// Function to access members based on index
	using base_type = typename value_traits<std::decay_t<T>>::type;
	using return_type = std::conditional_t<
	                    std::is_const_v<std::remove_reference_t<T>>,
	                    base_type const&,
	                    base_type&>;

	template <int N>
	static inline __host__ __device__ constexpr return_type at(T in)
	{
		if constexpr (std::is_same_v<std::decay_t<T>, float> || std::is_same_v<std::decay_t<T>, int8_t>) {
			// Handle the case for float and int8_t
			static_assert(N == 0, "Index out of bounds for float or int8_t");
			return in;
		} else {
			// Handle the case for types with x, y, z, w members
			if constexpr (N == 0) {
				return static_cast<return_type>(in.x);
			} else if constexpr (N == 1) {
				return static_cast<return_type>(in.y);
			} else if constexpr (N == 2) {
				return static_cast<return_type>(in.z);
			} else if constexpr (N == 3) {
				return static_cast<return_type>(in.w);
			} else {
				static_assert(N < 4, "Index out of bounds for type with x, y, z, w");
			}
		}
	}
};

// A generic struct to apply arguments to a static callable
template <int Index, StokesParameter S>
struct Invoker {
	template <typename Operator, typename... Args>
	static inline  __host__ __device__ void apply(Args&&... args)
	{
		Operator::template apply<Index, S>(std::forward<Args>(args)...);
	}
};

// Base case for recursion: no elements left
template <int Index, StokesParameter... S>
struct Iterate {
	template <typename Operator, typename... Args>
	static inline  __host__ __device__  void apply(Args&&... args)
	{
		// No-op when there are no more values
	}
};

// Recursive case: process the first element and recurse
template <int Index, StokesParameter First, StokesParameter... Rest>
struct Iterate<Index, First, Rest...> {
	template <typename Operator, typename... Args>
	static inline  __host__ __device__  void apply(Args&&... args)
	{
		Invoker<Index, First>::template apply<Operator>(
		    std::forward<Args>(args)...);
		Iterate<Index + 1, Rest...>::template apply<Operator>(
		    std::forward<Args>(args)...); // Recurse with the next element
	}
};


/**
 * Here we wrap the operations we wish to apply across the 
 * stokes parameters 
 */
struct IntegrateStokes {
	template <int I, StokesParameter S, typename T>
	static inline  __host__ __device__ void
	apply(float2 const& p0, float2 const& p1, T& power)
	{
		AT(power, I) += calculate_stokes<S>(p0, p1);
	}
};

struct IntegrateWeightedStokes {
	template <int I, StokesParameter S, typename T>
	static inline  __host__ __device__  void apply(float2 const& p0,
	                  float2 const& p1,
	                  T& power,
	                  float const& weight)
	{
		AT(power, I) += calculate_stokes<S>(p0, p1) * weight;
	}
};

struct IncoherentBeamSubtract {
	template <int I, StokesParameter S, typename T>
	static inline  __host__ __device__  void apply(T const& power,
	                  T const& ib_power,
	                  float const& ib_mutliplier, // 127^2 as default
	                  float const& scale_factor,
	                  T& result)
	{
		AT(result, I) = rintf((AT(power, I) - AT(ib_power, I) * ib_mutliplier) / scale_factor);
	}
};

struct Rescale {
	template <int I, StokesParameter S, typename T>
	static inline  __host__ __device__  void apply(T const& power,
	                  float const& offset,
	                  float const& scale_factor,
	                  T& result)
	{
		if constexpr(S == StokesParameter::I) {
			AT(result, I) = rintf((AT(power, I) - offset) / scale_factor);
		} else {
			AT(result, I) = rintf(AT(power, I) / scale_factor);
		}
	}
};

struct Clamp {
	template <int I, StokesParameter S, typename T, typename X>
	static inline  __host__ __device__ void apply(T const& power, X& result)
	{
		using EType = typename value_traits<X>::type;
		AT(result, I) = static_cast<EType>(
		                    fmaxf(static_cast<float>(
		                              std::numeric_limits<EType>::lowest()),
		                          fminf(static_cast<float>(
		                                    std::numeric_limits<EType>::max()),
		                                AT(power, I))));
	}
};

/**
 * Here we bring everything together to provide a fully generic StokesTraits implementation 
 */
template <StokesParameter... Stokes>
struct StokesTraits
{
	using RawPowerType       =  typename stokes_storage_type<sizeof...(Stokes)>::RawPowerType;
	using QuantisedPowerType =  typename stokes_storage_type<sizeof...(Stokes)>::QuantisedPowerType;
	constexpr static const RawPowerType zero_power = RawPowerType{};

	static inline __host__ __device__ void
	integrate_stokes(float2 const& p0,
	                 float2 const& p1,
	                 RawPowerType& power) {
		Iterate<0, Stokes...>::template apply<IntegrateStokes>(p0, p1, power);
	}

	static inline __host__ __device__ void
	integrate_weighted_stokes(float2 const& p0,
	                          float2 const& p1,
	                          RawPowerType& power,
	                          float const& weight) {
		Iterate<0, Stokes...>::template apply<IntegrateWeightedStokes>(p0, p1, power, weight);
	}

	static inline __host__ __device__ RawPowerType
	ib_subtract(RawPowerType const& power,
	            RawPowerType const& ib_power,
	            float const& ib_mutliplier,
	            float const& scale_factor) {
		RawPowerType result{};
		Iterate<0, Stokes...>::template apply<IncoherentBeamSubtract>(power, ib_power, ib_mutliplier, scale_factor, result);
		return result;
	}

	static inline __host__ __device__ RawPowerType
	rescale(RawPowerType const& power,
	        float const& offset,
	        float const& scale_factor) {
		RawPowerType result{};
		Iterate<0, Stokes...>::template apply<Rescale>(power, offset, scale_factor, result);
		return result;
	}

	static inline __host__ __device__ QuantisedPowerType
	clamp(RawPowerType const& power) {
		QuantisedPowerType result{};
		Iterate<0, Stokes...>::template apply<Clamp>(power, result);
		return result;
	}
};

// To provide back compatibility we will make typedefs for the existing types

template <StokesParameter Stokes>
using SingleStokesBeamformerTraits = StokesTraits<Stokes>;
using FullStokesBeamformerTraits = StokesTraits<I, Q, U, V>;


/**OLD code 
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
                float const& ib_mutliplier, // 127^2 as default
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
*/









} // namespace skyweaver

#endif // SKYWEAVER_BEAMFORMER_UTILS_CUH