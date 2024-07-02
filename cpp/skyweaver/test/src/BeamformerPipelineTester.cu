#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/beamformer_utils.cuh"
#include "skyweaver/IncoherentDedispersionPipeline.cuh"
#include "skyweaver/test/BeamformerPipelineTester.cuh"


#include <filesystem>
namespace fs = std::filesystem;

namespace skyweaver
{
namespace test
{
namespace {
    std::string default_dada_header = R"(
HEADER       DADA
HDR_VERSION  1.0
HDR_SIZE     4096
DADA_VERSION 1.0

# DADA parameters
OBS_ID       3

FILE_SIZE    1073745920
FILE_NUMBER  0

# time of the rising edge of the first time sample
# UTC_START    1708082482.092000
UTC_START    1708082170.0
MJD_START    60356.473172

OBS_OFFSET   0
OBS_OVERLAP  0

SOURCE       J1644-4559_Offset1
RA           16:44:26.25
DEC          -45:59:09.6
TELESCOPE    MeerKAT
INSTRUMENT   CBF-Feng
RECEIVER     L-band
FREQ         1284000000.000000
BW           856000000.0
TSAMP        0.0000047850467290

BYTES_PER_SECOND 3424000000.0

NBIT         8
NDIM         2
NPOL         2
NCHAN     64
NANT      57
ORDER    TAFTP
INNER_T  256

#MeerKAT specifics
SYNC_TIME    1708039531.000000
SAMPLE_CLOCK 1712000000.0
SAMPLE_CLOCK_START 0.0
CHAN0_IDX 2688

1708082168.957
)";
}


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

BeamformerPipelineTester::BeamformerPipelineTester(): ::testing::Test()
{
}

BeamformerPipelineTester::~BeamformerPipelineTester()
{
}

void BeamformerPipelineTester::SetUp()
{
}

void BeamformerPipelineTester::TearDown()
{
}

TEST_F(BeamformerPipelineTester, instantiate)
{
    using BfTraits = SingleStokesBeamformerTraits<StokesParameter::I>;
    PipelineConfig config;
    config.delay_file("data/test_delays.bin");
    NullHandler cb_handler;
    NullHandler ib_handler;
    NullHandler stats_handler;
    BeamformerPipeline<decltype(cb_handler),
                       decltype(ib_handler),
                       decltype(stats_handler),
                       BfTraits>(config, cb_handler, ib_handler, stats_handler);
}

TEST_F(BeamformerPipelineTester, full_pipeline_test)
{
    using BfTraits = SingleStokesBeamformerTraits<StokesParameter::I>;
    PipelineConfig config;
    config.delay_file("data/test_delays.bin");
    ObservationHeader header;
    std::vector<char> header_bytes(default_dada_header.size()+1);
    std::strcpy(header_bytes.data(), default_dada_header.c_str());
    psrdada_cpp::RawBytes raw_header(header_bytes.data(), default_dada_header.size(), default_dada_header.size(), false);
    read_dada_header(raw_header, header);
    validate_header(header, config);

    NullHandler cb_handler;
    NullHandler ib_handler;
    NullHandler stats_handler;
    using IDPipelineType = IncoherentDedispersionPipeline<typename BfTraits::QuantisedPowerType, typename BfTraits::QuantisedPowerType, decltype(cb_handler)>;
    using BPipelineType = BeamformerPipeline<IDPipelineType, decltype(ib_handler), decltype(stats_handler), BfTraits>;
    using InputVectorType = typename BPipelineType::HostVoltageVectorType;

    IDPipelineType dedispersion_pipeline(config, cb_handler);
    BPipelineType pipeline(config, dedispersion_pipeline, ib_handler, stats_handler);

    std::size_t input_size = header.nantennas * config.nchans() * config.npol() * config.gulp_length_samps();
    InputVectorType input(input_size);
    
    pipeline.init(header);
    for (int ii = 0; ii < 100; ++ii)
    {
        pipeline(input);
    }
}

} // namespace test
} // namespace skyweaver
