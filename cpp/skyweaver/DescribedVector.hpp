#ifndef SKYWEAVER_DESCRIBEDVECTORS_HPP
#define SKYWEAVER_DESCRIBEDVECTORS_HPP

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <sstream>
#include <initializer_list>
#include <type_traits>

namespace skyweaver
{

template <typename T, typename A>
inline std::ostream& operator<<(std::ostream& stream, std::vector<T, A> const& vec) {
    bool first = true;
    stream << "(";
    for (T const& val: vec)
    {
        if (!first){
            stream << ", ";
        } else {
            first = false;
        }
        stream << val;        
    }
    stream << ")";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, char2 const& val) {
    stream << "(" << static_cast<int>(val.x) 
    << "," << static_cast<int>(val.y) 
    << ")";
    return stream;
}

// Define the Dimension enum
enum Dimension { TimeDim, FreqDim, BeamDim, AntennaDim, PolnDim, DispersionDim };

std::string dimensions_to_string(std::vector<Dimension> const& dims);

// Helper to check if a Dimension is in a list of Dimensions
template <Dimension D, Dimension... Ds>
struct contains;

template <Dimension D>
struct contains<D> : std::false_type {};

template <Dimension D, Dimension Head, Dimension... Tail>
struct contains<D, Head, Tail...> 
    : std::conditional_t<D == Head, std::true_type, contains<D, Tail...>> {};

template <typename VectorType_, Dimension... dims> struct DescribedVector;

template <typename T, typename U>
struct is_dv_copyable : std::false_type {};

template <
    template <typename, typename> class Container1, 
    template <typename, typename> class Container2,
    typename T, 
    typename A1,
    typename A2,
    Dimension... dims
> struct is_dv_copyable<
    DescribedVector<Container1<T, A1>, dims...>, 
    DescribedVector<Container2<T, A2>, dims...>> 
: std::true_type {};

// The DescribedVector class
template <typename VectorType_, Dimension... dims>
struct DescribedVector {
    
    using VectorType = VectorType_;
    using value_type = typename VectorType::value_type;
    using FrequenciesType =  std::vector<double>;
    using DispersionMeasuresType =  std::vector<float>;
    
    DescribedVector()
    : _dms_stale(true), _frequencies_stale(true), _dims{dims...}, _tsamp(0.0) {
    }

    DescribedVector(std::initializer_list<std::size_t> sizes)
    : _dms_stale(true), _frequencies_stale(true), _sizes(sizes), _dims{dims...}, _tsamp(0.0) {
        if (_sizes.size() != sizeof...(dims)) {
            throw std::invalid_argument("Number of sizes must match number of dimensions");
        }
        _vector.resize(calculate_nelements());
    }
    
    ~DescribedVector(){};

    template <typename OtherDescribedVector, 
              typename = std::enable_if_t<is_dv_copyable<DescribedVector, OtherDescribedVector>::value>>
    DescribedVector(const OtherDescribedVector& other)
    : _sizes(other._sizes), _dims{other._dims}, 
      _vector(other._vector.begin(), other._vector.end()), 
      _dms(other._dms), _frequencies(other._frequencies), 
      _dms_stale(other._dms_stale), _tsamp(other._tsamp),
      _utc_offset(other._utc_offset), _reference_dm(other._reference_dm),
      _frequencies_stale(other._frequencies_stale) {
    }

    template <typename OtherDescribedVector, 
              typename = std::enable_if_t<is_dv_copyable<DescribedVector, OtherDescribedVector>::value>>
    void like(const OtherDescribedVector& other){
        resize(other._sizes);
        metalike(other);
    }

    template <typename OtherDescribedVector>
    void metalike(const OtherDescribedVector& other){
        _frequencies = other._frequencies;
        _frequencies_stale = other._frequencies_stale;
        _dms = other._dms;
        _dms_stale = other._dms_stale;
        _tsamp = other._tsamp;
        _utc_offset = other._utc_offset;
        _reference_dm = other._reference_dm;
    }

    auto& operator[](std::size_t idx)
    {
        return _vector[idx];
    }

    auto const& operator[](std::size_t idx) const
    {
        return _vector[idx];
    }

    void resize(std::initializer_list<std::size_t> sizes){
        _sizes = sizes;
        _vector.resize(calculate_nelements());
    }

    void resize(std::vector<std::size_t> const& sizes){
        _sizes = sizes;
        _vector.resize(calculate_nelements());
    }

    std::size_t size() const
    {
        return _vector.size();
    }

    auto data() noexcept
    {
        return _vector.data();
    }
    
    const auto data() const noexcept
    {
        return _vector.data();
    }

    auto begin() noexcept
    {
        return _vector.begin();
    }
    
    const auto begin() const noexcept
    {
        return _vector.begin();
    }

    auto end() noexcept
    {
        return _vector.end();
    }
    
    const auto end() const noexcept
    {
        return _vector.end();
    }

    VectorType const& vector() const{
        return _vector;
    }
    
    VectorType& vector(){
        return _vector;
    }

    std::size_t calculate_nelements() const {
        return std::accumulate(_sizes.begin(), _sizes.end(), 1, std::multiplies<>());
    }
    
    template <Dimension dim>
    std::size_t get_dim_extent() const {
        if constexpr (!contains<dim, dims...>::value)
        {
            return 1;
        } else {
            std::size_t total_size = 1;
            for (std::size_t i = 0; i < _dims.size(); ++i) {
                if (_dims[i] == dim) {
                    total_size *= _sizes[i];
                }
            }
            return total_size;
        }
    }
    
    FrequenciesType const& frequencies() const {
        if (_frequencies_stale)
        {
            throw std::runtime_error("Frequencies must be explicitly set before they can be used");
        }
         return _frequencies;
    }
    
    void frequencies(FrequenciesType const& freqs) {
        if (freqs.size() != get_dim_extent<FreqDim>())
        {
            throw std::runtime_error("Invalid number of frequecies passed.");
        }
        _frequencies_stale = false;
        _frequencies = freqs;
    }
    
    void frequencies(typename FrequenciesType::value_type const& freq) {
        if (get_dim_extent<FreqDim>() != 1)
        {
            throw std::runtime_error("Invalid number of frequecies passed.");
        }
        _frequencies_stale = false;
        _frequencies.resize(1, freq);
    }

    std::size_t nchannels() const {
         return get_dim_extent<FreqDim>();
    }
    
    std::size_t nantennas() const {
         return get_dim_extent<AntennaDim>();
    }
    
    std::size_t nsamples() const {
         return get_dim_extent<TimeDim>();
    }
    
    std::size_t nbeams() const {
         return get_dim_extent<BeamDim>();
    }    
    
    std::size_t npol() const {
         return get_dim_extent<PolnDim>();
    }  

    std::size_t ndms() const {
         return get_dim_extent<DispersionDim>();
    } 

    void tsamp(double tsamp_){
        _tsamp = tsamp_;
    }

    double tsamp() const {
        return _tsamp;
    }

    double utc_offset() const {
        return _utc_offset;
    }

    std::vector<std::size_t> const& extents() const
    {
        return _sizes;
    }

    DispersionMeasuresType const& dms() const {
        if (_dms_stale)
        {
             throw std::runtime_error("Dms must be explicitly set before they can be used");
         }
         return _dms;
    }
    
    void dms(DispersionMeasuresType const& dms) {
    if (dms.size() != get_dim_extent<DispersionDim>())
        {
            throw std::runtime_error("Invalid number of dispersion measures passed.");
        }
        _dms_stale = false;
        _dms = dms;
    }
    
    void dms(typename DispersionMeasuresType::value_type const& dm) {
    if (get_dim_extent<DispersionDim>() != 1)
        {
            throw std::runtime_error("Invalid number of dispersion measures passed.");
        }
        _dms_stale = false;
        _dms.resize(1);
        _dms[0] = dm;
    }

    void reference_dm(float dm){
       _reference_dm = dm;
    }

    float reference_dm() const {
        return _reference_dm;
    }

    std::string dims_as_string() const
    {
        return dimensions_to_string(_dims);
    }

    std::string describe() const
    {   
        std::stringstream stream;
        stream << "DescribedVector: " << dimensions_to_string(_dims) << " (" << size() << " elements)\n";
        stream << "  dimensions: " << _sizes << "\n";
        stream << "  nsamples: " << nsamples() << "\n";
        stream << "  nchans: " << nchannels() << "\n";
        stream << "  nantennas: " << nantennas() << "\n";
        stream << "  nbeams: " << nbeams() << "\n";
        stream << "  ndms: " << ndms() << "\n";
        stream << "  npol: " << npol() << "\n";
        stream << "  ndms: " << ndms() << "\n";
        stream << std::setprecision(15);
        stream << "  frequencies (Hz): " << _frequencies << "\n";
        stream << "  DMs (pc cm^-3): " << _dms << "\n";
        stream << "  Coherent DM (pc cm^-3): " << reference_dm() << "\n";
        stream << "  Time resolution (s): " << _tsamp << "\n";
        stream << "  Time offset (s): " << _utc_offset << "\n";
        stream << "  Stale frequencies: " << _frequencies_stale << "\n";
        stream << std::setprecision(6);
        return stream.str();
    }

    DispersionMeasuresType _dms;
    bool _dms_stale;
    FrequenciesType _frequencies;
    bool _frequencies_stale;
    std::vector<std::size_t> _sizes;
    std::vector<Dimension> _dims;
    double _tsamp = 0.0;
    double _utc_offset = 0.0;
    float _reference_dm = 0.0;
    VectorType _vector;
};

template <typename T>
struct is_device_vector : std::false_type {};

template <typename T, typename Alloc>
struct is_device_vector<thrust::device_vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc, Dimension... Dims>
struct is_device_vector<DescribedVector<thrust::device_vector<T, Alloc>, Dims...>> : std::true_type {};

// Pipeline inputs
template <typename T> using TAFTPVoltagesH = DescribedVector<thrust::host_vector<T>, TimeDim, AntennaDim, FreqDim, TimeDim, PolnDim>;
template <typename T> using TAFTPVoltagesD = DescribedVector<thrust::device_vector<T>, TimeDim, AntennaDim, FreqDim, TimeDim, PolnDim>;
// Beamformer inputs
template <typename T> using FTPAVoltagesH = DescribedVector<thrust::host_vector<T>, FreqDim, TimeDim, PolnDim, AntennaDim>;
template <typename T> using FTPAVoltagesD = DescribedVector<thrust::device_vector<T>, FreqDim, TimeDim, PolnDim, AntennaDim>;
// Coherent dedisperser inputs
template <typename T> using TPAVoltagesH = DescribedVector<thrust::host_vector<T>, TimeDim, PolnDim, AntennaDim>;
template <typename T> using TPAVoltagesD = DescribedVector<thrust::device_vector<T>, TimeDim, PolnDim, AntennaDim>;
// Coherent beamformer outputs
template <typename T> using TFBPowersH = DescribedVector<thrust::host_vector<T>, TimeDim, FreqDim, BeamDim>;
template <typename T> using TFBPowersD = DescribedVector<thrust::device_vector<T>, TimeDim, FreqDim, BeamDim>;
// Incoherent beamformer outputs
template <typename T> using BTFPowersH = DescribedVector<thrust::host_vector<T>, BeamDim, TimeDim, FreqDim>;
template <typename T> using BTFPowersD = DescribedVector<thrust::device_vector<T>, BeamDim, TimeDim, FreqDim>;
// Incoherent dedisperser outputs
template <typename T> using TDBPowersH = DescribedVector<thrust::host_vector<T>, TimeDim, DispersionDim, BeamDim>;
template <typename T> using TDBPowersD = DescribedVector<thrust::device_vector<T>, TimeDim, DispersionDim, BeamDim>;
// Statistics outputs
template <typename T> using FPAStatsH = DescribedVector<thrust::host_vector<T>, FreqDim, PolnDim, AntennaDim>;
template <typename T> using FPAStatsD = DescribedVector<thrust::device_vector<T>, FreqDim, PolnDim, AntennaDim>;

} // namespace skyweaver

#endif //SKYWEAVER_DESCRIBEDVECTORS_HPP