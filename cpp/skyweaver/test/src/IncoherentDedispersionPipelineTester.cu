#include "skyweaver/test/IncoherentDedispersionPipelineTester.cuh"
#include "skyweaver/IncoherentDedispersionPipeline.cuh"
#include "skyweaver/DescribedVector.hpp"
#include <cstring>

namespace skyweaver
{
namespace test
{

template <typename Traits>
IncoherentDedispersionPipelineTester<Traits>::IncoherentDedispersionPipelineTester()
: _init_called(false), _operator_call_count(0)
{
}

template <typename Traits>
IncoherentDedispersionPipelineTester<Traits>::~IncoherentDedispersionPipelineTester(){
}

template <typename Traits>
void IncoherentDedispersionPipelineTester<Traits>::SetUp(){
    _config.centre_frequency(1284e6);
    _config.bandwidth(13375000.0);
    _config.coherent_dms({0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
}

template <typename Traits>
void IncoherentDedispersionPipelineTester<Traits>::TearDown(){
}

template <typename Traits>
void IncoherentDedispersionPipelineTester<Traits>::init(ObservationHeader const& header)
{
    _init_called = true;
    _init_arg = header;
}

template <typename Traits>
void IncoherentDedispersionPipelineTester<Traits>::operator()(OutputVectorType const& output, std::size_t dm_idx)
{
    ++_operator_call_count;
    _operator_arg_1 = output;
    _operator_arg_2 = dm_idx;
}

typedef ::testing::Types<
    IDPipelineTraits<int8_t, int8_t>,
    IDPipelineTraits<char4, char4>,
    > TestTypes;

TYPED_TEST_SUITE(IncoherentDedispersionPipelineTester, TestTypes);

TYPED_TEST(IncoherentDedispersionPipelineTester, init_call) {
    using Traits = TypeParam;
    using IType = typename Traits::InputType;
    using OType = typename Traits::OutputType;
    using Handler = decltype(*this);
    IncoherentDedispersionPipeline<IType, OType, Handler> pipeline(this->_config, *this);
    ObservationHeader header;
    pipeline.init(header);
    ASSERT_TRUE(this->_init_called);
    // Note this is not a safe way to compare structs, but should work for this 
    // specific case.
    ASSERT_TRUE(std::memcmp(&(this->_init_arg), &header, sizeof(header)));
    ASSERT_EQ(this->_operator_call_count, 0);
}

TYPED_TEST(IncoherentDedispersionPipelineTester, operator_calls) {
    using Traits = TypeParam;
    using IType = typename Traits::InputType;
    using OType = typename Traits::OutputType;
    using Handler = decltype(*this);
    IncoherentDedispersionPipeline<IType, OType, Handler> pipeline(this->_config, *this);
    ObservationHeader header;
    pipeline.init(header);
    auto const& dms = this->_config.coherent_dms();
    TFBPowersH<IType> input({8192, this->_config.nchans(), this->_config.nbeams()});
    for (int ii = 0; ii < 2; ++ii){
        for (int dm_idx = 0; dm_idx < dms.size(); ++dm_idx)
        {
            pipeline(input, dm_idx);
        }
    }
    ASSERT_GT(this->_operator_call_count, 0);
}

} // namespace test
} // namespace skyweaver
