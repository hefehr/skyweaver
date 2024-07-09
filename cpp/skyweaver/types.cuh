#ifndef SKYWEAVER_TYPES_CUH
#define SKYWEAVER_TYPES_CUH

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include <iostream>

// Operator overloading for CUDA vector types
template <typename T>
struct is_vec4: std::false_type {
};

template <>
struct is_vec4<float4>: std::true_type {
};

template <>
struct is_vec4<char4>: std::true_type {
};

template <typename T>
inline constexpr bool is_vec4_v = is_vec4<T>::value;

template <typename T>
struct value_traits {};

template <>
struct value_traits<int8_t>
{
    typedef int8_t type;
    typedef float promoted_type;
    __host__ __device__ static constexpr int8_t zero() { return 0; }
    __host__ __device__ static constexpr int8_t one() { return 1; }
};

template <>
struct value_traits<float>
{
    typedef float type;
    typedef float promoted_type;
    __host__ __device__ static constexpr float zero() { return 0.0f; }
    __host__ __device__ static constexpr float one() { return 1.0f; }
};

template <>
struct value_traits<char4>
{
    typedef int8_t type;
    typedef float4 promoted_type;
    __host__ __device__ static constexpr char4 zero() { return {0, 0, 0, 0}; }
    __host__ __device__ static constexpr char4 one() { return {1, 1, 1, 1}; }
};


template <>
struct value_traits<float4> {
    typedef float type;
    typedef float4 promoted_type;
    __host__ __device__ static constexpr float4 zero() { return {0.0f, 0.0f, 0.0f, 0.0f}; }
    __host__ __device__ static constexpr float4 one() { return {1.0f, 1.0f, 1.0f, 1.0f}; }
};

inline std::ostream& operator<<(std::ostream& stream, char4 const& val) {
    stream << "(" << static_cast<int>(val.x) 
    << "," << static_cast<int>(val.y) 
    << "," << static_cast<int>(val.z) 
    << "," << static_cast<int>(val.w) 
    << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, float4 const& val) {
    stream << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")";
    return stream;
}

/**
 * vector - vector operations
 * explicit static_casts used to avoid Wnarrowing errors for int8_t types due to integral promotion 
 * (over/underflow is the expected behaviour here).
 */
template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, T>::type operator+(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x + rhs.x), 
            static_cast<typename value_traits<T>::type>(lhs.y + rhs.y),
            static_cast<typename value_traits<T>::type>(lhs.z + rhs.z),
            static_cast<typename value_traits<T>::type>(lhs.w + rhs.w)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, T>::type operator+=(T& lhs, const X& rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, T>::type operator-(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x - rhs.x), 
            static_cast<typename value_traits<T>::type>(lhs.y - rhs.y),
            static_cast<typename value_traits<T>::type>(lhs.z - rhs.z),
            static_cast<typename value_traits<T>::type>(lhs.w - rhs.w)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, T>::type operator*(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x * rhs.x), 
            static_cast<typename value_traits<T>::type>(lhs.y * rhs.y),
            static_cast<typename value_traits<T>::type>(lhs.z * rhs.z),
            static_cast<typename value_traits<T>::type>(lhs.w * rhs.w)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, T>::type operator/(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x / rhs.x), 
            static_cast<typename value_traits<T>::type>(lhs.y / rhs.y),
            static_cast<typename value_traits<T>::type>(lhs.z / rhs.z),
            static_cast<typename value_traits<T>::type>(lhs.w / rhs.w)};
}

template <typename T, typename X>
__host__ __device__ inline  typename std::enable_if<is_vec4_v<T> && is_vec4_v<X>, bool>::type operator==(const T& lhs, const X& rhs) {
    return  (lhs.x == rhs.x) && 
            (lhs.y == rhs.y) &&
            (lhs.z == rhs.z) &&
            (lhs.w == rhs.w);
}

/**
 * vector - scalar operations
 */
template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator*(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x * rhs), 
            static_cast<typename value_traits<T>::type>(lhs.y * rhs),
            static_cast<typename value_traits<T>::type>(lhs.z * rhs),
            static_cast<typename value_traits<T>::type>(lhs.w * rhs)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator+=(const T& lhs, const X& rhs) {
    lhs.x += rhs;
    lhs.y += rhs;
    lhs.z += rhs;
    lhs.w += rhs;
    return lhs;
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator/(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x / rhs), 
            static_cast<typename value_traits<T>::type>(lhs.y / rhs),
            static_cast<typename value_traits<T>::type>(lhs.z / rhs),
            static_cast<typename value_traits<T>::type>(lhs.w / rhs)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator+(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x + rhs), 
            static_cast<typename value_traits<T>::type>(lhs.y + rhs),
            static_cast<typename value_traits<T>::type>(lhs.z + rhs),
            static_cast<typename value_traits<T>::type>(lhs.w + rhs)};
}

template <typename T, typename X>
__host__ __device__ inline typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator-(const T& lhs, const X& rhs) {
    return {static_cast<typename value_traits<T>::type>(lhs.x - rhs), 
            static_cast<typename value_traits<T>::type>(lhs.y - rhs),
            static_cast<typename value_traits<T>::type>(lhs.z - rhs),
            static_cast<typename value_traits<T>::type>(lhs.w - rhs)};

}

template <typename U, typename T>
static inline typename std::enable_if<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, U>::type clamp(T const& value)
{
    return static_cast<U>(
        fmaxf(static_cast<float>(
                    std::numeric_limits<U>::lowest()),
                fminf(static_cast<float>(
                        std::numeric_limits<U>::max()),
                    static_cast<float>(value))));
}

template <typename U, typename T>
static inline typename std::enable_if<is_vec4_v<T> && is_vec4_v<U>, U>::type clamp(T const& value)
{
    U clamped;
    clamped.x = clamp<typename value_traits<U>::type, typename value_traits<T>::type>(value.x);
    clamped.y = clamp<typename value_traits<U>::type, typename value_traits<T>::type>(value.y);
    clamped.z = clamp<typename value_traits<U>::type, typename value_traits<T>::type>(value.z);
    clamped.w = clamp<typename value_traits<U>::type, typename value_traits<T>::type>(value.w);
    return clamped;
}

#endif //SKYWEAVER_TYPES_CUH