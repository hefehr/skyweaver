#ifndef SKYWEAVER_NVTX_HELPERS
#define SKYWEAVER_NVTX_HELPERS

#ifdef USE_NVTX
    #include <nvToolsExt.h>
    #define NVTX_MARKER(msg) nvtxMark(msg)
    #define NVTX_RANGE_PUSH(msg) nvtxRangePush(msg)
    #define NVTX_RANGE_POP() nvtxRangePop()
#else
    #define NVTX_MARKER(msg) 
    #define NVTX_RANGE_PUSH(msg)
    #define NVTX_RANGE_POP()
#endif

#endif //SKYWEAVER_NVTX_HELPERS