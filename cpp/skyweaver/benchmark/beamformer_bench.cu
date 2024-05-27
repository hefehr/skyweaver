#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/CoherentBeamformer.cuh"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/skyweaver_constants.hpp"

#include <cuda.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
static void BM_CustomCounter(benchmark::State& state)
{
    int64_t num_operations = 0;

    for(auto _: state) {
        // Simulate some work
        for(int i = 0; i < state.range(0); ++i) {
            ++num_operations;
            benchmark::DoNotOptimize(num_operations);
        }
    }

    // Use counter to report the number of operations
    state.counters["Operations"] = benchmark::Counter(num_operations);
    state.counters["Operations/Second"] =
        benchmark::Counter(num_operations, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_CustomCounter)->ArgsProduct({{1000, 2000, 3000}})->MinTime(1.0);

void BM_LOOP2(benchmark::State& state)
{
    int n = state.range(0);
    int z = 0;
    for(auto _: state) {
        for(int ii = 0; ii < n; ++ii) { z += ii; }
        benchmark::DoNotOptimize(z);
    }
    state.counters["rate"] = benchmark::Counter(n, benchmark::Counter::kIsRate);
    state.counters["n"]    = benchmark::Counter(n);
}

BENCHMARK(BM_LOOP2)
    ->ArgsProduct({{10, 100, 1000, 10000, 100000}})
    ->Iterations(3000);

void BM_LOOP(benchmark::State& state)
{
    int n = state.range(0);
    for(auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(n));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start);
        state.SetIterationTime(time_span.count());
    }
    state.counters["rate"] =
        benchmark::Counter(n, benchmark::Counter::kAvgIterationsRate);
    state.counters["n"] = n;
}

BENCHMARK(BM_LOOP)
    ->ArgsProduct({{10, 100, 1000}})
    ->Iterations(3)
    ->UseManualTime();

void BM_COHERENT_BEAMFORMER(benchmark::State& state)
{
    cudaStream_t _stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
    skyweaver::PipelineConfig _config;
    skyweaver::CoherentBeamformer coherent_beamformer(_config);
    skyweaver::IncoherentBeamformer incoherent_beamformer(_config);
    std::size_t ntimestamps = max(
        1L,
        SKYWEAVER_CB_PACKET_SIZE / (_config.nchans() / _config.cb_fscrunch()) /
            (_config.nsamples_per_heap() / _config.cb_tscrunch()));
    ntimestamps =
        max(ntimestamps,
            SKYWEAVER_CB_NSAMPLES_PER_BLOCK / _config.nsamples_per_heap());
    std::size_t input_size =
        (ntimestamps * _config.nantennas() * _config.nchans() *
         _config.nsamples_per_heap() * _config.npol());
    std::size_t weights_size =
        _config.nantennas() * _config.nchans() * _config.nbeams();
    skyweaver::CoherentBeamformer::VoltageVectorType ftpa_voltages_gpu(
        input_size);
    skyweaver::CoherentBeamformer::WeightsVectorType fbpa_weights_gpu(
        weights_size);
    skyweaver::CoherentBeamformer::PowerVectorType btf_powers_gpu;
    skyweaver::CoherentBeamformer::ScalingVectorType scales(
        _config.nchans());
    skyweaver::CoherentBeamformer::ScalingVectorType offsets(
        _config.nchans());
    skyweaver::CoherentBeamformer::PowerVectorType tf_powers_gpu;
    skyweaver::CoherentBeamformer::RawPowerVectorType tf_powers_raw_gpu;
    for(auto _: state) {
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
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

BENCHMARK(BM_COHERENT_BEAMFORMER)->Iterations(30);

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
