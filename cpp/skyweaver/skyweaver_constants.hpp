#ifndef SKYWEAVER_CONSTANTS_HPP
#define SKYWEAVER_CONSTANTS_HPP

/**
 * COMIPLE TIME PARAMETERS FOR SKYWEAVER
 *
 * The defines below specify compile time constants for skyweaver.
 * For performance reasons, the code should be re-compiled for the
 * specific input required. This can be done by editing the contents
 * of this file or, preferably, by passing the parameters for these
 * defines to cmake during the build process, e.g.
 *
 * cmake -S . -B build/ -DSKYWEAVER_NANTENNAS=48 -DSKYWEAVER_NBEAMS=192
 *
 */


/**
 * The number of antennas to use for beamforming
 *
 * Skyweaver uses specifies only one set of antennas to
 * beamform. To specify subsets for coherent beamforming
 * scalar antenna weightings can be set in the delay model.
 *
 * This parameter needs to be set to a multiple of 4. If input
 * data is passed with few antennas the input will be padded up
 * to this value.
 */
#ifndef SKYWEAVER_NANTENNAS
    #define SKYWEAVER_NANTENNAS 64
#endif // SKYWEAVER_NANTENNAS
static_assert(SKYWEAVER_NANTENNAS % 4 == 0,
              "SKYWEAVER_NBEAMS must be a multiple of 4");

/**
 * The number of frequency channels in the input data set.
 *
 * TODO: Check if this can be removed from these hardcoded params
 *       and if it can be instead read from file.
 */
#ifndef SKYWEAVER_NCHANS
    #define SKYWEAVER_NCHANS 64
#endif // SKYWEAVER_NCHANS

/**
 * The total number of coherent beams to produce.
 * The number of beams must be a multiple of 32.
 *
 */
#ifndef SKYWEAVER_NBEAMS
    #define SKYWEAVER_NBEAMS 32
#endif // SKYWEAVER_NBEAMS
static_assert(SKYWEAVER_NBEAMS % 32 == 0,
              "SKYWEAVER_NBEAMS must be a multiple of 32");

/**
 * Enable IB subtraction from the CB.
 */
#ifndef SKYWEAVER_IB_SUBTRACTION
    #define SKYWEAVER_IB_SUBTRACTION 1
#endif // SKYWEAVER_IB_SUBTRACTION

/**
 * The factor by which to time scrunch the 
 * coherent beam data.
 */
#ifndef SKYWEAVER_CB_TSCRUNCH
    #define SKYWEAVER_CB_TSCRUNCH 16
#endif // SKYWEAVER_CB_TSCRUNCH

/**
 * The factor by which to time scrunch the 
 * coherent beam data.
 */
#ifndef SKYWEAVER_CB_FSCRUNCH
    #define SKYWEAVER_CB_FSCRUNCH 4
#endif // SKYWEAVER_CB_FSCRUNCH

/**
 * The factor by which to time scrunch the 
 * incoherent beam data.
 */
#ifndef SKYWEAVER_IB_TSCRUNCH
    #define SKYWEAVER_IB_TSCRUNCH 16
#endif // SKYWEAVER_IB_TSCRUNCH

/**
 * The factor by which to time scrunch the 
 * incoherent beam data.
 */
#ifndef SKYWEAVER_IB_FSCRUNCH
    #define SKYWEAVER_IB_FSCRUNCH 4
#endif // SKYWEAVER_IB_FSCRUNCH


// These are fixed for MeerKAT F-engine data
#define SKYWEAVER_NSAMPLES_PER_HEAP 256
#define SKYWEAVER_NPOL              2

//A useful number to compute is the size of each AFTP in TAFTP input data
// Usually for A=64, N=64, T=256, P=2, this is 8192 bytes
#define SKYWEAVER_INPUT_NBITS sizeof(std::int8_t)
#define SKYWEAVER_AFTP_SIZE \
                SKYWEAVER_NANTENNAS * \
                SKYWEAVER_NCHANS * \
                SKYWEAVER_NSAMPLES_PER_HEAP * \
                SKYWEAVER_NPOL * \
                SKYWEAVER_INPUT_NBITS

// These parameters are fixed for beamformer 
// kernel performance. 
// Note: The old FBFUSE code contained parameters
// here for specifying spead heap and packet sizes
// for the outgoing data. With skyweaver it is assumed
// that the data do not need to be packetised and it is 
// preferable to output in a format more suitable for 
// downstream processing without network transfer.
#define SKYWEAVER_CB_NTHREADS  1024
#define SKYWEAVER_CB_WARP_SIZE 32
#define SKYWEAVER_CB_NWARPS_PER_BLOCK \
    (SKYWEAVER_CB_NTHREADS / SKYWEAVER_CB_WARP_SIZE)
#define SKYWEAVER_CB_NSAMPLES_PER_BLOCK \
    (SKYWEAVER_CB_TSCRUNCH * SKYWEAVER_CB_NTHREADS / SKYWEAVER_CB_WARP_SIZE)
#define SKYWEAVER_CB_NCHANS_OUT        (SKYWEAVER_NCHANS / SKYWEAVER_CB_FSCRUNCH)

// To be removed when the new output format is chosen
#define SKYWEAVER_CB_PACKET_SIZE 8192     
#define SKYWEAVER_CB_HEAP_SIZE 8192
#define SKYWEAVER_CB_NCHANS_PER_PACKET (SKYWEAVER_CB_NCHANS_OUT)
#define SKYWEAVER_CB_NSAMPLES_PER_PACKET (SKYWEAVER_CB_HEAP_SIZE / SKYWEAVER_CB_NCHANS_OUT)
#define SKYWEAVER_CB_NPACKETS_PER_HEAP (SKYWEAVER_CB_HEAP_SIZE / SKYWEAVER_CB_PACKET_SIZE)
#define SKYWEAVER_CB_NSAMPLES_PER_HEAP (SKYWEAVER_CB_NPACKETS_PER_HEAP * SKYWEAVER_CB_NSAMPLES_PER_PACKET)


#define SKYWEAVER_IB_NSAMPLES_PER_BLOCK \
    (SKYWEAVER_IB_TSCRUNCH * SKYWEAVER_IB_NTHREADS / SKYWEAVER_IB_WARP_SIZE)
#define SKYWEAVER_IB_NCHANS_OUT        (SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH)

// To be removed
#define SKYWEAVER_IB_PACKET_SIZE 8192     
#define SKYWEAVER_IB_HEAP_SIZE 8192
#define SKYWEAVER_IB_NCHANS_PER_PACKET (SKYWEAVER_IB_NCHANS_OUT)
#define SKYWEAVER_IB_NSAMPLES_PER_PACKET (SKYWEAVER_IB_HEAP_SIZE / SKYWEAVER_IB_NCHANS_OUT)
#define SKYWEAVER_IB_NPACKETS_PER_HEAP (SKYWEAVER_IB_HEAP_SIZE / SKYWEAVER_IB_PACKET_SIZE)
#define SKYWEAVER_IB_NSAMPLES_PER_HEAP (SKYWEAVER_IB_NPACKETS_PER_HEAP * SKYWEAVER_IB_NSAMPLES_PER_PACKET)

#endif // SKYWEAVER_CONSTANTS_HPP

