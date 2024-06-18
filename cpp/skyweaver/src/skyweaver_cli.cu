#include "boost/program_options.hpp"
#include "errno.h"
#include "psrdada_cpp/cli_utils.hpp"
#include "skyweaver/BeamformerPipeline.cuh"
#include "skyweaver/MultiFileReader.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "thrust/host_vector.h"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <vector>

#define BOOST_LOG_DYN_LINK 1

namespace
{
const size_t ERROR_IN_COMMAND_LINE     = 1;
const size_t SUCCESS                   = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

class NullHandler
{
  public:
    template <typename... Args>
    void init(Args... args) {};

    template <typename... Args>
    bool operator()(Args... args)
    {
        return false;
    };
};
} // namespace

// This patching of the << operator is required to allow
// for float vector arguments to boost program options
namespace std
{
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec)
{
    for(auto item: vec) { os << item << " "; }
    return os;
}
} // namespace std

template <typename BfTraits>
void run_pipeline(skyweaver::PipelineConfig const& config);

int main(int argc, char** argv)
{
    try {
        skyweaver::PipelineConfig config;

        /**
         * Define and parse the program options
         */
        namespace po = boost::program_options;

        // Generic options group here to contain the configuration file name
        // The config file will be parsed AFTER the the command line options
        // this gives command line options precedence over config file options.
        // Options here are only settable via the command line.
        po::options_description generic("Generic options");
        generic.add_options()("cfg,c",
                              po::value<std::string>()->default_value(""),
                              "Skyweaver configuration file");

        // Main option group that contains parameters settable via both the
        // command line and the config file
        po::options_description main_options("Main options");
        main_options.add_options()

            // Help menu
            ("help,h", "Display help messages")

            // Input file containing list of DADA files to process
            ("input-file",
             po::value<std::string>()->required()->notifier(
                 [&config](std::string key) {
                     config.read_input_file_list(key);
                 }),
             "File containing list of DADA files to process")

            // Input file for delay solutions
            // This can contain any number of beams but only beams
            // up to the maximum configured for generation will be
            // produced. Antenna ordering in the file must match
            // the antenna order of the input data.
            ("delay-file",
             po::value<std::string>()->required()->notifier(
                 [&config](std::string key) { config.delay_file(key); }),
             "File containing delay solutions")

            // Output file for block statistics
            ("stats-file",
             po::value<std::string>()
                 ->default_value(config.statistics_file())
                 ->notifier([&config](std::string key) {
                     config.statistics_file(key);
                 }),
             "Output file for block statistics")

            // Output directory where all results will be written
            ("output-dir",
             po::value<std::string>()
                 ->default_value(config.output_dir())
                 ->notifier(
                     [&config](std::string key) { config.output_dir(key); }),
             "The output directory for all results")

            // Output file for block statistics
            ("output-level",
             po::value<float>()
                 ->default_value(config.output_level())
                 ->notifier([&config](float key) { config.output_level(key); }),
             "The desired standard deviation for output data")

            /**
             * Dispersion measures for coherent dedispersion
             * Can be specified on the command line with:
             *
             * --coherent-dm 1 2 3
             * or
             * --coherent-dm 1 --coherent-dm 2 --coherent-dm 3
             *
             * In the configuration file it can only be specified with:
             *
             * coherent-dm=1
             * coherent-dm=2
             * coherent-dm=3
             */
            ("coherent-dm",
             po::value<std::vector<float>>()
                 ->multitoken()
                 ->default_value(config.coherent_dms())
                 ->notifier([&config](std::vector<float> const& dms) {
                     config.coherent_dms(dms);
                 }),
             "The dispersion measures to coherently dedisperse to")
            // Number of samples to read in each gulp
            ("gulp-size",
             po::value<std::size_t>()
                 ->default_value(config.gulp_length_samps())
                 ->notifier([&config](std::size_t const& gulp_size) {
                     // Round off to next multiple of 256
                     if(gulp_size % config.nsamples_per_heap() != 0) {
                         BOOST_LOG_TRIVIAL(debug)
                             << "Rounding up gulp-size to next multiple of 256";
                         config.gulp_length_samps(
                             (gulp_size / config.nsamples_per_heap()) *
                             config.nsamples_per_heap());
                     } else {
                         config.gulp_length_samps(gulp_size);
                     }
                 }),
             "The number of samples to read in each gulp ")

            // Stokes mode I, Q, U, V or IQUV
            ("stokes-mode",
             po::value<std::string>()->default_value("I")->notifier(
                 [&config](std::string stokes) {
                     for(auto& c: stokes) c = (char)toupper(c);
                     config.stokes_mode(stokes);
                 }),
             "The Stokes mode to use, can be either I, Q, U, V or IQUV")

            // Logging options
            ("log-level",
             po::value<std::string>()->default_value("info")->notifier(
                 [](std::string level) { psrdada_cpp::set_log_level(level); }),
             "The logging level to use (debug, info, warning, error)");

        // set options allowed on command line
        po::options_description cmdline_options;
        cmdline_options.add(generic).add(main_options);

        // set options allowed in config file
        po::options_description config_file_options;
        config_file_options.add(main_options);

        po::variables_map variable_map;
        try {
            po::store(po::command_line_parser(argc, argv)
                          .options(cmdline_options)
                          .run(),
                      variable_map);
            if(variable_map.count("help")) {
                std::cout << "skyweavercpp -- C++/CUDA beamformer pipeline for "
                             "COMPACT-ERC"
                          << std::endl
                          << cmdline_options << std::endl;
                return SUCCESS;
            }
        } catch(po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

        auto config_file = variable_map.at("cfg").as<std::string>();

        if(config_file != "") {
            std::ifstream config_fs(config_file.c_str());
            if(!config_fs.is_open()) {
                std::cerr << "Unable to open configuration file: "
                          << config_file << " (" << std::strerror(errno)
                          << ")\n";
                return ERROR_UNHANDLED_EXCEPTION;
            } else {
                po::store(po::parse_config_file(config_fs, config_file_options),
                          variable_map);
            }
        }
        po::notify(variable_map);

        /**
         * All the application code goes here
         */
        BOOST_LOG_TRIVIAL(info)
            << "Initialising the skyweaver beamforming pipeline";
        if(config_file != "") {
            BOOST_LOG_TRIVIAL(info) << "Configuration file: " << config_file;
        }
        BOOST_LOG_TRIVIAL(info)
            << "Input file count: " << config.input_files().size();
        BOOST_LOG_TRIVIAL(info) << "Delay file: " << config.delay_file();
        BOOST_LOG_TRIVIAL(info) << "Stats file: " << config.statistics_file();
        BOOST_LOG_TRIVIAL(info) << "Output dir: " << config.output_dir();
        BOOST_LOG_TRIVIAL(info) << "Output level: " << config.output_level();
        BOOST_LOG_TRIVIAL(info) << "Coherent DMs: " << config.coherent_dms();
        BOOST_LOG_TRIVIAL(info) << "Gulp size: " << config.gulp_length_samps();
        if(config.stokes_mode() == "I") {
            run_pipeline<skyweaver::SingleStokesBeamformerTraits<
                skyweaver::StokesParameter::I>>(config);
        } else if(config.stokes_mode() == "Q") {
            run_pipeline<skyweaver::SingleStokesBeamformerTraits<
                skyweaver::StokesParameter::Q>>(config);
        } else if(config.stokes_mode() == "U") {
            run_pipeline<skyweaver::SingleStokesBeamformerTraits<
                skyweaver::StokesParameter::U>>(config);
        } else if(config.stokes_mode() == "V") {
            run_pipeline<skyweaver::SingleStokesBeamformerTraits<
                skyweaver::StokesParameter::V>>(config);
        } else if(config.stokes_mode() == "IQUV") {
            run_pipeline<skyweaver::FullStokesBeamformerTraits>(config);
        } else {
            throw std::runtime_error("Invalid Stokes mode passed, must be one "
                                     "of I, Q, U, V or IQUV");
        }
    } catch(std::exception& e) {
        std::cerr << "Unhandled Exception reached the top of main: " << e.what()
                  << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;
}

template <typename BfTraits>
void run_pipeline(skyweaver::PipelineConfig const& config)
{
    // Here we build and invoke the pipeline
    NullHandler cb_handler;
    NullHandler ib_handler;
    NullHandler stats_handler;
    skyweaver::MultiFileReader file_reader(config);

    skyweaver::BeamformerPipeline<decltype(cb_handler),
                                  decltype(ib_handler),
                                  decltype(stats_handler),
                                  BfTraits>
        pipeline(config, cb_handler, ib_handler, stats_handler);
    thrust::host_vector<char2> taftp_input_voltage;
    auto const& header = file_reader.get_header();

    // Calculate input size per gulp
    std::size_t input_elements = header.nantennas * config.nchans() *
                                 config.npol() * config.gulp_length_samps();
    taftp_input_voltage.resize(input_elements);
    std::size_t input_bytes =
        input_elements * sizeof(decltype(taftp_input_voltage)::value_type);

    if(config.nchans() != header.nchans) {
        std::runtime_error("Data has invalid number of channels");
    }
    pipeline.init(header);
    BOOST_LOG_TRIVIAL(info)
        << "Total input size (bytes): " << file_reader.get_total_size();

    // TODO: Add a parameter to PipelineConfig for start sample? time?
    // TODO: Add a parameter to PipelineConfig for nsamples? duration?
    int count = 3;
    while(!file_reader.eof()) {
        std::streamsize nbytes_read =
            file_reader.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(
                                 taftp_input_voltage.data())),
                             input_bytes);
        pipeline(taftp_input_voltage);
        --count;
        if(count == 0) {
            break;
        }
    }
}