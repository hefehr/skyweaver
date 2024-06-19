#ifndef SKYWEAVER_TYPES_CUH
#define SKYWEAVER_TYPES_CUH

namespace skyweaver 
{

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

/**
 * vector - vector operations
 */
template <typename T>
__host__ __device__ typename std::enable_if<is_vec4_v<T>, T>::type operator+(const T& lhs, const T& rhs) {
    return {lhs.x + rhs.x, 
            lhs.y + rhs.y,
            lhs.z + rhs.z,
            lhs.w + rhs.w};
}

template <typename T>
__host__ __device__ typename std::enable_if<is_vec4_v<T>, T>::type operator-(const T& lhs, const T& rhs) {
    return {lhs.x - rhs.x, 
            lhs.y - rhs.y,
            lhs.z - rhs.z,
            lhs.w - rhs.w};
}

template <typename T>
__host__ __device__ typename std::enable_if<is_vec4_v<T>, T>::type operator*(const T& lhs, const T& rhs) {
    return {lhs.x * rhs.x, 
            lhs.y * rhs.y,
            lhs.z * rhs.z,
            lhs.w * rhs.w};
}

template <typename T>
__host__ __device__ typename std::enable_if<is_vec4_v<T>, T>::type operator/(const T& lhs, const T& rhs) {
    return {lhs.x / rhs.x, 
            lhs.y / rhs.y,
            lhs.z / rhs.z,
            lhs.w / rhs.w};
}

/**
 * vector - scalar operations
 */
template <typename T, typename X>
__host__ __device__ typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator*(const T& lhs, const X& rhs) {
    return {lhs.x * rhs, 
            lhs.y * rhs,
            lhs.z * rhs,
            lhs.w * rhs};
}

template <typename T, typename X>
__host__ __device__ typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator/(const T& lhs, const X& rhs) {
    return {lhs.x / rhs, 
            lhs.y / rhs,
            lhs.z / rhs,
            lhs.w / rhs};
}

template <typename T, typename X>
__host__ __device__ typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator+(const T& lhs, const X& rhs) {
    return {lhs.x + rhs, 
            lhs.y + rhs,
            lhs.z + rhs,
            lhs.w + rhs};
}

template <typename T, typename X>
__host__ __device__ typename std::enable_if<is_vec4_v<T> && std::is_arithmetic_v<X>, T>::type operator-(const T& lhs, const X& rhs) {
    return {lhs.x - rhs, 
            lhs.y - rhs,
            lhs.z - rhs,
            lhs.w - rhs};

}



} //namespace skyweaver

#endif //SKYWEAVER_TYPES_CUH