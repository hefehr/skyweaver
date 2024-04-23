#ifndef PSRDADA_CPP_DADABLOCK_CPP
#define PSRDADA_CPP_DADABLOCK_CPP

#include "common.hpp"
#include <memory>
namespace psradada_cpp {

    class DADABlock {
    private:
        std::unique_ptr<thrust::host_vector<char>> h_aftp_db;
        std::unique_ptr<thrust::device_vector<char>> d_aftp_db;

    public:
        DADABlock();
        ~DADABlock();

         

    };
}

#endif