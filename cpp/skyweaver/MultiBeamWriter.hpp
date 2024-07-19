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

struct Bridge
{
  private:
      using  PowerType = std::vector<int8_t>;
  public:
    std::vector<std::string> _tdb_filenames;
    std::string freq;

}; // class Bridge

class MultiBeamWriter
{
  private:
    using FreqType = unsigned int; // up to the nearest Hz
    std::map<FreqType, std::unique_ptr<Bridge>> _bridges;
    std::vector<std::string> _beam_filenames;


        public
        : add_bridge(FreqType freq, std::vector<std::string> tdb_filenames);
    remove_bridge(FreqType freq);

    init();

} // class MultiFileWriter
} // namespace skyweaver