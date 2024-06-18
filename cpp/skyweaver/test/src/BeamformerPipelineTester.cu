#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/beamformer_utils.cuh"
#include "skyweaver/test/BeamformerPipelineTester.cuh"

#include <filesystem>
namespace fs = std::filesystem;

namespace skyweaver
{
namespace test
{

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
    PipelineConfig config;
    config.delay_file("data/test_delays.bin");
    NullHandler cb_handler;
    NullHandler ib_handler;
    NullHandler stats_handler;
    typedef SingleStokesBeamformerTraits<StokesParameter::I> BfTraits;
    BeamformerPipeline<decltype(cb_handler),
                       decltype(ib_handler),
                       decltype(stats_handler),
                       BfTraits>(config, cb_handler, ib_handler, stats_handler);
}

} // namespace test
} // namespace skyweaver
