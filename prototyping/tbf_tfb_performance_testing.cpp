/* 
Data ordering testing to determine preferred format for dedispersion

TFB is STRONGLY preffered as it leads to sequential memory access and 
thus optimal caching. Additionally iterating the data in this format
allows for the exploitation of 256-bit register SIMD instructions.

The code can be compiled with:

g++ -O3 -fopenmp tbf_tfb_performance_testing.cpp

Assembly can be dumped with:

g++ -O3 -march=native -ftree-vectorize -S tbf_tfb_performance_testing.cpp

*/



#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>
#include <numeric>
#include <omp.h>

void dedisperse_tbf(std::vector<float> const& tbf, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int nchans,
                    int nbeams,
                    int max_delay)
{
    const int nsamples = tbf.size() / (nbeams * nchans);
    const int bf = nbeams * nchans;

    for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
    {
        const int t = t_idx * nbeams;
        for (int b_idx = 0; b_idx < nbeams; ++b_idx)
        {
            float power = 0.0f;
            const int b = b_idx * nchans;
            for (int f_idx = 0; f_idx < nchans; ++f_idx)
            {
                int input_idx = (t_idx + delays[f_idx]) * bf + b + f_idx;
                power += tbf[input_idx];
            }
            tb[t + b_idx] = power;
        }
    }
}

void dedisperse_tbf_loop_reversed(std::vector<float> const& tbf, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int nchans,
                    int nbeams,
                    int max_delay)
{
    const int nsamples = tbf.size() / (nbeams * nchans);
    const int bf = nbeams * nchans;
    std::vector<float> powers(nbeams);

    for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
    {
        std::fill(powers.begin(), powers.end(), 0.0f);
        const int t = t_idx * nbeams;
        for (int f_idx = 0; f_idx < nchans; ++f_idx)
        {
            float power = 0.0f;
            const int t = (t_idx + delays[f_idx]) * bf + f_idx;
            for (int b_idx = 0; b_idx < nbeams; ++b_idx)
            {
                powers[b_idx] += tbf[t + b_idx * nchans];
            }
        }
        std::copy(powers.begin(), powers.end(), tb.begin() + t_idx * nbeams);
    }
}

void dedisperse_tfb(std::vector<float> const& tfb, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int nchans,
                    int nbeams,
                    int max_delay)
{
    const int nsamples = tfb.size() / (nbeams * nchans);
    const int bf = nbeams * nchans;
    std::vector<float> powers(nbeams);

    for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
    {
        //std::memset(powers.data(), 0.0f, powers.size() * sizeof(float));
        std::fill(powers.begin(), powers.end(), 0.0f);

        for (int f_idx = 0; f_idx < nchans; ++f_idx)
        {
            int idx = (t_idx + delays[f_idx]) * bf + f_idx * nbeams;
            for (int b_idx = 0; b_idx < nbeams; ++b_idx)
            {
                powers[b_idx] += tfb[idx + b_idx];
            }
        }
        std::copy(powers.begin(), powers.end(), tb.begin() + t_idx * nbeams);
    }
}

void dedisperse_tfb_no_vec(std::vector<float> const& tfb, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int nchans,
                    int nbeams,
                    int max_delay)
{
    const int nsamples = tfb.size() / (nbeams * nchans);
    const int bf = nbeams * nchans;
    for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
    {
        for (int f_idx = 0; f_idx < nchans; ++f_idx)
        {
            int idx = (t_idx + delays[f_idx]) * bf + f_idx * nbeams;
            for (int b_idx = 0; b_idx < nbeams; ++b_idx)
            {
                tb[t_idx * nbeams + b_idx] += tfb[idx + b_idx];
            }
        }
    }
}

#define NBEAMS 800
#define NCHANS 64

void dedisperse_tfb_hardcoded(std::vector<float> const& tfb, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int max_delay)
{
    const int nsamples = tfb.size() / (NBEAMS * NCHANS);
    const int bf = NBEAMS * NCHANS;
    std::vector<float> powers(NBEAMS);

    for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
    {
        std::fill(powers.begin(), powers.end(), 0.0f);
        for (int f_idx = 0; f_idx < NCHANS; ++f_idx)
        {
            int idx = (t_idx + delays[f_idx]) * bf + f_idx * NBEAMS;
            for (int b_idx = 0; b_idx < NBEAMS; ++b_idx)
            {
                powers[b_idx] += tfb[idx + b_idx];
            }
        }
        std::copy(powers.begin(), powers.end(), tb.begin() + t_idx * NBEAMS);
    }
}

void dedisperse_tfb_threaded(std::vector<float> const& tfb, 
                    std::vector<float>& tb, 
                    std::vector<int> const& delays,
                    int nchans,
                    int nbeams,
                    int max_delay)
{
    const int nsamples = tfb.size() / (nbeams * nchans);
    const int bf = nbeams * nchans;

    #pragma omp parallel
    {
        std::vector<float> powers(nbeams, 0.0f); // Initialize once per thread

        #pragma omp for
        for (int t_idx = 0; t_idx < (nsamples - max_delay); ++t_idx)
        {
            std::fill(powers.begin(), powers.end(), 0.0f);

            for (int f_idx = 0; f_idx < nchans; ++f_idx)
            {
                int idx_base = (t_idx + delays[f_idx]) * bf + f_idx * nbeams;
                for (int b_idx = 0; b_idx < nbeams; ++b_idx)
                {
                    powers[b_idx] += tfb[idx_base + b_idx];
                }
            }

            std::copy(powers.begin(), powers.end(), tb.begin() + t_idx * nbeams);
        }
    }
}

int main()
{
    float fbottom = 856.0f;
    float bw = 856.0f;
    float chbw = bw / 4096;
    int nchans = 64;
    float dm = 250.0;
    float tsamp = 16e-6;


    std::vector<float> freq(nchans);
    for (int ii=0; ii < nchans; ++ii)
    {
        freq[ii] = fbottom + ii * chbw;
    }

    std::vector<int> delays(64);
    for (int ii = 0; ii < delays.size(); ++ii)
    {
        float delay = (4.15e3 * dm * ((1.0 / (fbottom * fbottom)) - (1.0 / (freq[ii] * freq[ii]))) / tsamp);
        std::cout << "CHAN " << ii << " delay: " << delay << "\n";
        delays[ii] = static_cast<int>(delay + 0.5f);
    }

    int max_delay = delays.back();
    int nsamples = 8192;
    int nbeams = NBEAMS;

    std::vector<float> tbf((nsamples + max_delay) * nbeams * nchans, 1);

    {
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tbf(tbf, tb, delays, nchans, nbeams, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tbf: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }

    {
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tbf_loop_reversed(tbf, tb, delays, nchans, nbeams, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tbf_loop_reversed: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }
    
    {
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tfb(tbf, tb, delays, nchans, nbeams, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tfb: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }

        {
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tfb_no_vec(tbf, tb, delays, nchans, nbeams, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tfb_no_vec: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }

    {
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tfb_hardcoded(tbf, tb, delays, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tfb_hardcoded: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }
    
    {
        omp_set_num_threads(16);
        std::vector<float> tb(nsamples * nbeams, 0.0f);
        auto start = std::chrono::high_resolution_clock::now();
        dedisperse_tfb_threaded(tbf, tb, delays, nchans, nbeams, max_delay);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by dedisperse_tfb_threaded: " << duration.count() << " milliseconds" << std::endl;
        float sum = std::accumulate(tb.begin(), tb.end(), 0);
        std::cout << sum << "\n";
    }

    return 0;
}