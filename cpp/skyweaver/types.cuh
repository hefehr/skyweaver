#ifndef SKYWEAVER_TYPES_CUH
#define SKYWEAVER_TYPES_CUH

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include <iostream>

// Operator overloading for CUDA vector types


template <typename T>
struct is_vecN : std::disjunction<
                    std::is_same<T, float2>,
                    std::is_same<T, float3>,    
                    std::is_same<T, float4>,
                    std::is_same<T, char2>,
                    std::is_same<T, char3>,
                    std::is_same<T, char4>
                > {};

template <typename T>
inline constexpr bool is_vecN_v = is_vecN<T>::value;

template <typename T>
struct value_traits {
};

template <>
struct value_traits<int8_t> {
    typedef int8_t type;
    typedef float promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 1; }
    __host__ __device__ static constexpr int8_t zero() { return 0; }
    __host__ __device__ static constexpr int8_t one() { return 1; }
};

template <>
struct value_traits<char2> {
    typedef int8_t type;
    typedef float2 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 2; }
    __host__ __device__ static constexpr char2 zero() { return {0, 0}; }
    __host__ __device__ static constexpr char2 one() { return {1, 1}; }
};

template <>
struct value_traits<char3> {
    typedef int8_t type;
    typedef float3 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 3; }
    __host__ __device__ static constexpr char3 zero() { return {0, 0, 0}; }
    __host__ __device__ static constexpr char3 one() { return {1, 1, 1}; }
};

template <>
struct value_traits<char4> {
    typedef int8_t type;
    typedef float4 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 4; }
    __host__ __device__ static constexpr char4 zero() { return {0, 0, 0, 0}; }
    __host__ __device__ static constexpr char4 one() { return {1, 1, 1, 1}; }
};

template <>
struct value_traits<float> {
    typedef float type;
    typedef float promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 1; }
    __host__ __device__ static constexpr float zero() { return 0.0f; }
    __host__ __device__ static constexpr float one() { return 1.0f; }
};

template <>
struct value_traits<float2> {
    typedef float type;
    typedef float2 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 2; }
    __host__ __device__ static constexpr float2 zero()
    {
        return {0.0f, 0.0f};
    }
    __host__ __device__ static constexpr float2 one()
    {
        return {1.0f, 1.0f};
    }
};

template <>
struct value_traits<float3> {
    typedef float type;
    typedef float3 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 3; }
    __host__ __device__ static constexpr float3 zero()
    {
        return {0.0f, 0.0f, 0.0f};
    }
    __host__ __device__ static constexpr float3 one()
    {
        return {1.0f, 1.0f, 1.0f};
    }
};

template <>
struct value_traits<float4> {
    typedef float type;
    typedef float4 promoted_type;
    __host__ __device__ static constexpr int8_t size() { return 4; }
    __host__ __device__ static constexpr float4 zero()
    {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    __host__ __device__ static constexpr float4 one()
    {
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }
};

inline std::ostream& operator<<(std::ostream& stream, char2 const& val)
{
    stream << "(" << static_cast<int>(val.x) << "," << static_cast<int>(val.y)
           << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, char3 const& val)
{
    stream << "(" << static_cast<int>(val.x) << "," << static_cast<int>(val.y)
           << "," << static_cast<int>(val.z) << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, char4 const& val)
{
    stream << "(" << static_cast<int>(val.x) << "," << static_cast<int>(val.y)
           << "," << static_cast<int>(val.z) << "," << static_cast<int>(val.w)
           << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, float2 const& val)
{
    stream << "(" << val.x << "," << val.y << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, float3 const& val)
{
    stream << "(" << val.x << "," << val.y << "," << val.z << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, float4 const& val)
{
    stream << "(" << val.x << "," << val.y << "," << val.z << "," << val.w
           << ")";
    return stream;
}

/**
 * vector - vector operations
 * explicit static_casts used to avoid Wnarrowing errors for int8_t types due to
 * integral promotion (over/underflow is the expected behaviour here).
 */


template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, T>::type operator+(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x + rhs.x);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y + rhs.y);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z + rhs.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w + rhs.w);
    }
    return r;
}


template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, T>::type
    operator+=(T& lhs, const X& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    if constexpr (value_traits<T>::size() > 2){
        lhs.z += rhs.z;
    }
    if constexpr (value_traits<T>::size() > 3){
        lhs.w += rhs.w;
    }
    return lhs;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, T>::type
    operator-(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x - rhs.x);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y - rhs.y);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z - rhs.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w - rhs.w);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, T>::type
    operator*(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x * rhs.x);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y * rhs.y);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z * rhs.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w * rhs.w);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, T>::type
    operator/(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x / rhs.x);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y / rhs.y);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z / rhs.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w / rhs.w);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<X>, bool>::type
    operator==(const T& lhs, const X& rhs)
{
    bool r = (lhs.x == rhs.x) && (lhs.y == rhs.y);
    if constexpr (value_traits<T>::size() > 2){
        r = r && (lhs.z == rhs.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        r = r && (lhs.w == rhs.w);
    }
    return r;
}

/**
 * vector - scalar operations
 */
template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && std::is_arithmetic_v<X>, T>::type
    operator*(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x * rhs);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y * rhs);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z * rhs);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w * rhs);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && std::is_arithmetic_v<X>, T>::type
    operator+=(const T& lhs, const X& rhs)
{
    lhs.x += rhs;
    lhs.y += rhs;
    if constexpr (value_traits<T>::size() > 2){
        lhs.z += rhs;
    }
    if constexpr (value_traits<T>::size() > 3){
        lhs.w += rhs;
    }
    return lhs;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && std::is_arithmetic_v<X>, T>::type
    operator/(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x / rhs);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y / rhs);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z / rhs);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w / rhs);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && std::is_arithmetic_v<X>, T>::type
    operator+(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x + rhs);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y + rhs);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z + rhs);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w + rhs);
    }
    return r;
}

template <typename T, typename X>
__host__ __device__ inline
    typename std::enable_if<is_vecN_v<T> && std::is_arithmetic_v<X>, T>::type
    operator-(const T& lhs, const X& rhs)
{
    T r;
    r.x = static_cast<typename value_traits<T>::type>(lhs.x - rhs);
    r.y = static_cast<typename value_traits<T>::type>(lhs.y - rhs);
    if constexpr (value_traits<T>::size() > 2){
        r.z = static_cast<typename value_traits<T>::type>(lhs.z - rhs);
    }
    if constexpr (value_traits<T>::size() > 3){
        r.w = static_cast<typename value_traits<T>::type>(lhs.w - rhs);
    }
    return r;
}

template <typename U, typename T>
__host__ __device__ static inline
    typename std::enable_if<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                            U>::type
    clamp(T const& value)
{
    return static_cast<U>(
        fmaxf(static_cast<float>(std::numeric_limits<U>::lowest()),
              fminf(static_cast<float>(std::numeric_limits<U>::max()),
                    static_cast<float>(value))));
}

template <typename U, typename T>
__host__ __device__ static inline
    typename std::enable_if<is_vecN_v<T> && is_vecN_v<U>, U>::type
    clamp(T const& value)
{
    U clamped;
    clamped.x =
        clamp<typename value_traits<U>::type, typename value_traits<T>::type>(
            value.x);
    clamped.y =
        clamp<typename value_traits<U>::type, typename value_traits<T>::type>(
            value.y);
    if constexpr (value_traits<T>::size() > 2){
        clamped.z =
            clamp<typename value_traits<U>::type, typename value_traits<T>::type>(
                value.z);
    }
    if constexpr (value_traits<T>::size() > 3){
        clamped.w =
            clamp<typename value_traits<U>::type, typename value_traits<T>::type>(
                value.w);
    }
    return clamped;
}

#endif // SKYWEAVER_TYPES_CUH