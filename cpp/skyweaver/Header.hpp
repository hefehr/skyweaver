#ifndef SKYWEAVER_HEADER_HPP
#define SKYWEAVER_HEADER_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

namespace skyweaver
{

/**
 * @brief      A helper class for managing DADA headers
 *
 * @detail     DADA headers are composed of ASCII key-value
 *             pairs stored in a single char array.
 */
class Header
{
  public:
    /**
     * @brief      Constructs and instance of the Header class
     *
     * @param      header  A RawBytes object wrapping a DADA header
     */
    explicit Header(psrdada_cpp::RawBytes& header);
    ~Header();
    Header(Header const&) = delete;

    /**
     * @brief      Get a value from the header
     *
     * @param      key   An ASCII key (the name of the value to get)
     *
     * @tparam     T     The data type of the parameter to be read
     *
     * @return     The value corresponding to the given key
     */
    template <typename T>
    T get(char const* key) const;

    /**
     * @brief      Set a value in the header
     *
     * @param      key    An ASCII key (the name of the value to be set)
     * @param[in]  value  The value to set
     *
     * @tparam     T      The type of value being set
     */
    template <typename T>
    void set(char const* key, T value);

    /**
     * @brief      Clear a DADA header
     *
     * @details    Memsets the entire buffer to zero
     */
    void purge();

  private:
    void fetch_header_string(char const* key) const;

  private:
    psrdada_cpp::RawBytes& _header;
    char _buffer[1024];
};

} // namespace skyweaver

#endif // SKYWEAVER_HEADER_HPP
