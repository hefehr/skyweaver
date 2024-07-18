#include "SigprocFileWriter.hpp"
#include "boost/log/trivial.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "skyweaver/ObservationHeader.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace skyweaver
{

class Bridge
{
  public:
    typedef std::vector<int8_t> PowerType;
    std::vector<std::string> tdb_filenames;
    std::vector<std::string> beam_filenames;

}; // class Bridge

class MultiBeamWriter
{
  private:
    std::vector<unsigned int> freqs;
    std::vector<std::unique_ptr<Bridge>> bridges;
    std::vector<bool>

        public
        : add_bridge(FreqType freq, std::vector<std::string> tdb_filenames);
    remove_bridge(FreqType freq);

    init();

} // class MultiFileWriter
} // namespace skyweaver