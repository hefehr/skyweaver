#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/CoherentBeamformer.cuh"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/skyweaver_constants.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <thread>

class BeamformerBencher: public benchmark::Fixture
{
  public:
    void SetUp(::benchmark::State const& state)
    {
        std::size_t ntimestamps = state.range(0); // ntimestamps
        std::size_t input_size =
            (ntimestamps * _config.nantennas() * _config.nchans() *
             _config.nsamples_per_heap() * _config.npol());
        std::size_t weights_size =
            _config.nantennas() * _config.nchans() * _config.nbeams();
        ftpa_voltages_gpu.resize(input_size);
        fbpa_weights_gpu.resize(weights_size);
        scales.resize(_config.nchans() / _config.cb_fscrunch());
        offsets.resize(_config.nchans() / _config.cb_fscrunch());
        CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
        CUDA_ERROR_CHECK(cudaEventCreate(&start));
        CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    }

    void TearDown(::benchmark::State const& state)
    {
        CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
        CUDA_ERROR_CHECK(cudaEventDestroy(start));
        CUDA_ERROR_CHECK(cudaEventDestroy(stop));
    }

    // Implemented to avoid warning #611-D
    void SetUp(::benchmark::State& state)
    {
        SetUp(const_cast<const ::benchmark::State&>(state));
    }
    void TearDown(::benchmark::State& state)
    {
        TearDown(const_cast<const ::benchmark::State&>(state));
    }

  public:
    cudaStream_t _stream;
    cudaEvent_t start, stop;
    skyweaver::PipelineConfig _config;
    skyweaver::CoherentBeamformer::VoltageVectorType ftpa_voltages_gpu;
    skyweaver::CoherentBeamformer::WeightsVectorType fbpa_weights_gpu;
    skyweaver::CoherentBeamformer::PowerVectorType btf_powers_gpu;
    skyweaver::CoherentBeamformer::ScalingVectorType scales;
    skyweaver::CoherentBeamformer::ScalingVectorType offsets;
    skyweaver::CoherentBeamformer::PowerVectorType tf_powers_gpu;
    skyweaver::CoherentBeamformer::RawPowerVectorType tf_powers_raw_gpu;
};

BENCHMARK_DEFINE_F(BeamformerBencher, simple_bench)(benchmark::State& state)
{
    typedef skyweaver::CoherentBeamformer::VoltageVectorType::value_type
        ValueType;
    float elapsed_time;
    float total_elapsed_time = 0.0f;
    skyweaver::CoherentBeamformer coherent_beamformer(_config);
    skyweaver::IncoherentBeamformer incoherent_beamformer(_config);
    for(auto _: state) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, 0));
        for(int ii = 0; ii < 10; ++ii) {
            incoherent_beamformer.beamform(ftpa_voltages_gpu,
                                           tf_powers_raw_gpu,
                                           tf_powers_gpu,
                                           scales,
                                           offsets,
                                           _stream);
            coherent_beamformer.beamform(ftpa_voltages_gpu,
                                         fbpa_weights_gpu,
                                         scales,
                                         offsets,
                                         tf_powers_raw_gpu,
                                         btf_powers_gpu,
                                         _stream);
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
        CUDA_ERROR_CHECK(cudaEventRecord(stop, 0));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        state.SetIterationTime(elapsed_time);
        total_elapsed_time += elapsed_time;
    }
    state.counters["Ntimestamps"] = state.range(0);
    float performance = (ftpa_voltages_gpu.size() * sizeof(ValueType) *
                         state.iterations() * 10) /
                        total_elapsed_time / 1e9;
    state.counters["Performance GB/s"] = performance;
}

BENCHMARK_REGISTER_F(BeamformerBencher, simple_bench)
    ->Args({1, 10, 100})
    ->Iterations(30);

int main(int argc, char** argv)
{
    char arg0_default[] = "benchmark";
    char* args_default  = arg0_default;
    psrdada_cpp::set_log_level("INFO");
    if(!argv) {
        argc = 1;
        argv = &args_default;
    }
    ::benchmark::Initialize(&argc, argv);
    if(::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
