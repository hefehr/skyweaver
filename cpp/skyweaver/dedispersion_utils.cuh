#ifndef SKYWEAVER_DEDISPERSION_UTILS_CUH
#define SKYWEAVER_DEDISPERSION_UTILS_CUH

namespace skyweaver
{

template <typename T>
static constexpr T pi = static_cast<T>(3.1415926535897932385);
template <typename T>
static constexpr T two_pi = static_cast<T>(2 * pi<T>);
template <typename T>
static constexpr T dm_const_hz_s =
    static_cast<T>(4.148806423e15); // Hz^2 pc^-1 cm^3 s
template <typename T>
static constexpr T dm_const_mhz_us =
    static_cast<T>(4.148806423e9); // MHz^2 pc^-1 cm^3 us

// Freq in Hz!
template <typename T>
__host__ __device__ T dm_delay(T f_ref, T f, T dm)
{
    return static_cast<T>(dm_const_hz_s<T> *
                          (1 / (f_ref * f_ref) - 1 / (f * f)) * dm);
}

struct DMSampleDelay {
    __host__ __device__ DMSampleDelay(double dm, double f_ref, double tsamp)
        : _dm(dm), _f_ref(f_ref), _tsamp(tsamp)
    {
    }

    __host__ __device__ int operator()(double const& freq)
    {
        return static_cast<int>(dm_delay<double>(_f_ref, freq, _dm) / _tsamp +
                                0.5);
    }

    double _dm;
    double _f_ref; // reference frequency Hz
    double _tsamp; // sampling interval seconds
};

struct DMPrefactor {
    __host__ __device__ double operator()(double const& dm)
    {
        return -1.0f * two_pi<double> * dm_const_hz_s<double> * dm;
    }
};

} // namespace skyweaver

#endif // SKYWEAVER_DEDISPERSION_UTILS_CUH
