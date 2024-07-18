import numpy as np
nant = 4
nchan = 16
nsamples = 10000
npol = 2
data = np.random.randint(0, 255, size=nsamples*nchan*nant*npol*2, dtype=np.uint8)
MJD = 60356.47024305579093
tsamp = 4.7850467290e-6

data_header = '''
HEADER       DADA
HDR_VERSION  1.0
HDR_SIZE     4096
DADA_VERSION 1.0

# DADA parameters
OBS_ID       test

FILE_SIZE    956305408
FILE_NUMBER  0

# time of the rising edge of the first time sample
UTC_START    1708082229.000020336
MJD_START    60356.47024305579093



SOURCE       J1644-4559
RA           16:44:49.27
DEC          -45:59:09.7
TELESCOPE    MeerKAT
INSTRUMENT   CBF-Feng
RECEIVER     L-band
FREQ         856000000.000000
BW           13.375
TSAMP        4.7850467290

BYTES_PER_SECOND 3424000000.000000

OBS_NCHAN   4096
OBS_BW      856000000.000000
OBS_FREQ    1284000000.000000
OBS_OFFSET   0
OBS_OVERLAP  0

NBIT         8
NDIM         2
NPOL         2
NCHAN        16
NANT         4
ORDER        TAFTP
INNER_T      256

#MeerKAT specifics
SYNC_TIME    1708039531.000000
SAMPLE_CLOCK 1712000000.000000
SAMPLE_CLOCK_START 73098976034816
CHAN0_IDX 2688
'''
data_header_raw_bytes = data_header.encode()

data_header_size = len(data_header_raw_bytes)
#fill the rest upto 4096 bytes with null
data_header_raw_bytes += b'\0'*(4096 - data_header_size)
with open('../../cpp/skyweaver/test/data/dada_files/data.dada', 'wb') as f:
    f.write(data_header_raw_bytes)
    f.write(data.tobytes())

    
#split data into 10 chunks
chunk_size = len(data)//10
print( "Generating chunks of size", chunk_size)
for i in range(10):
    chunk_header= data_header.replace('OBS_OFFSET   0', f'OBS_OFFSET   {i*chunk_size}')
    data_header_raw_bytes = chunk_header.encode()
    chunk_data_header_size = len(data_header_raw_bytes)
    data_header_raw_bytes += b'\0'*(4096 - chunk_data_header_size)

    with open(f'../../cpp/skyweaver/test/data/dada_files/data_chunk_{i}.dada', 'wb') as f:
        f.write(data_header_raw_bytes)
        f.write(data[i*chunk_size:(i+1)*chunk_size].tobytes())

print("Done")
