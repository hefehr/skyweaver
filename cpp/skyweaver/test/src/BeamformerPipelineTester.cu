#include "skyweaver/PipelineConfig.hpp"
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
    NullHandler cb_handler;
    NullHandler ib_handler;
    NullHandler stats_handler;
    BeamformerPipeline<decltype(cb_handler),
                       decltype(ib_handler),
                       decltype(stats_handler)>(config,
                                                cb_handler,
                                                ib_handler,
                                                stats_handler);
}

} // namespace test
} // namespace skyweaver
