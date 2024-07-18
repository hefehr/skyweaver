import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc

DM_CONST=4.148806423e9
def dm_delay(nu1, nu2, dm, tsamp):
   return ((1/pow(nu1/1000,2.0)) - (1/pow(nu2/1000, 2.0)))*dm*0.004148806423/tsamp

# create TAFTP voltages

nant =  64
npol = 2
ncomplex = 2
nsamples = 1024
nbits = 8
dm  = 1000

cfreq = 1284 #MHz
total_bw = 856 # MHz
total_nchans = 4096
coarse_chan_width = total_bw / total_nchans


nbridges = 64
nchans_per_bridge = int(total_nchans / nbridges)
bridge_bw = total_bw / nbridges

ibridge = 0

tsamp = 4096/856e6
fftlen = nsamples

print("central_f0: ", cfreq)
print("bridge_bw: ", bridge_bw)
print("tsamp: ", tsamp)
print("dm: ", dm)
print("nchans_per_bridge: ", nchans_per_bridge)
print("chanwidth: ", coarse_chan_width)



flow_ibridge = cfreq + (ibridge - nbridges/2) * bridge_bw
print("flow_ibridge: ", flow_ibridge)

f0_ibridge = flow_ibridge + bridge_bw/2
print("f0_ibridge: ", f0_ibridge)

channels = flow_ibridge + coarse_chan_width * np.arange(0, nchans_per_bridge)
chanedges = flow_ibridge - coarse_chan_width/2.0  + coarse_chan_width*np.arange(0, nchans_per_bridge+1)



print("channels: ", channels)
print("chanedges: ", chanedges)


chirps = np.zeros((nchans_per_bridge, fftlen),dtype='complex128')
chirp_chan_width = coarse_chan_width/fftlen
print("chirp_chan_width: ", chirp_chan_width)


def chirp(flow, df, dm_prefactor):
   phase = (df * df * dm_prefactor)/((flow + df) * flow * flow)
   
   #print("************", flow, df[1023], dm_prefactor, df[1023] * df[1023] * dm_prefactor, phase[1023], np.exp(1j*phase[1023]))
   sini = np.sin(phase)
   cosi = np.cos(phase)
   return np.array(cosi + 1j*sini).astype(np.complex128)


fine_chans = np.arange(fftlen) * chirp_chan_width
for k in range(len(fine_chans)):
   print("fine_chans[{}]: {}".format(k, flow_ibridge + fine_chans[k]))
print("fine_chans: ",   fine_chans)
dm_prefactor = -2*np.pi*dm*DM_CONST
print("dm_prefactor: ", dm_prefactor)

for i in range(nchans_per_bridge):
   chirps[i] = chirp(chanedges[i], fine_chans, dm_prefactor)

chirps[0].tofile("/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion/chirp_python.bin")


print(np.shape(chirps))

   

print("chirps: ", chirps.shape)
# for c in chirps:
#    print(c[0:20])
#    print(np.max(np.angle(c))*180.0/np.pi)

print(chirps[0][0:20])
print(chirps[0][-20:])

print(np.max(np.angle(chirps[0]))*180.0/np.pi)


print("computing the max shift from the dm...")
#generate max shifts for all channels
max_shift = int(dm_delay(chanedges[0], chanedges[1],dm, tsamp))
print("max_shift: ", max_shift)
max_shift = max_shift + 1 if max_shift%2 != 0 else max_shift

max_shift_tpa = max_shift * nant * npol;
print("max_shift_tpa: ", max_shift_tpa)

block_length_tpa = nsamples * nant * npol
print("block_length_tpa: ", block_length_tpa)



# create a random array of data
#data = np.random.randint(-1 * 2**nbits/2, 2**nbits/2 -1, (nant, npol, nchans_per_bridge, nsamples, ncomplex)).astype(np.int8)
data = np.random.randint(-1 * 2**nbits/2, 2**nbits/2 -1, (nchans_per_bridge, nsamples,npol,  nant, ncomplex)).astype(np.int8)

#print size in bytes
print("data size in bytes: ", data.nbytes)
print("data shape: ", data.shape)
# transposed_data = data.transpose(2,3,1,0,4)
# transposed_data[0].tofile("/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion/codedisp_input_DM{:03}.bin".format(dm))
# print(data.shape, transposed_data.shape)
data.tofile("/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion/codedisp_input_DM{:03}.bin".format(dm))

truncated_length = fftlen - max_shift

chunks = int(nsamples/truncated_length)

print("chunks: ", chunks)

print(truncated_length, fftlen, max_shift)
discard_size = int(max_shift/2) 


output = np.zeros((nchans_per_bridge, nsamples - max_shift, npol,  nant, ncomplex), dtype = np.int8)
print("output shape: ", output.shape)

print("Starting dedispersion....")
print("truncated_length: ", truncated_length)
first = True

for ant in range(nant):
   for pol in range(npol):
      first = True
      for i in range(chunks):
         print("ant: ", ant, " pol: ", pol, " chunk: ", i)
         start = 0 if first else i*truncated_length
         #d = data[ant,  pol, :, start : start + fftlen,:] 
         d = data[:, start : start + fftlen,  pol, ant, :] 
         print("start: ", start, " end: ", start + fftlen, "d.shape: ", d.shape)

         if(d.shape[1] != fftlen):
            print("Padding the data:",fftlen-d.shape[1])
            temp = np.zeros((nchans_per_bridge, fftlen - d.shape[1], ncomplex), dtype = np.int8)
            d = np.concatenate((d, temp), axis=1)

         d = d[:,:,0] + 1j*d[:,:,1]
         dedispersed_data = np.fft.ifft(np.fft.fft(d.astype(np.complex64), axis=1) * chirps, axis=1).astype(np.complex64)

              
         
         real = dedispersed_data[:,discard_size : fftlen - discard_size].real.round().astype(np.int8)
         imag = dedispersed_data[:,discard_size : fftlen - discard_size].imag.round().astype(np.int8)

         if((i+1)*truncated_length > (nsamples - max_shift)):
            print("last chunk")
            output[ :,  i*truncated_length:(nsamples - max_shift), pol, ant, 0] = real[:,  0:(nsamples - max_shift-i*truncated_length)]
            output[ :,  i*truncated_length:(nsamples - max_shift),pol, ant, 1] =  imag[:,  0:(nsamples - max_shift-i*truncated_length)]
         else:
            output[ :,  i*truncated_length:(i+1)*truncated_length, pol, ant, 0] = real 
            output[ :,  i*truncated_length:(i+1)*truncated_length, pol, ant, 1] =  imag
         first = False



#docker cp d2b81ed015c6:/tmp/codedisp_input.dat ~/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion
#docker cp d2b81ed015c6:/tmp/codedisp_output.dat ~/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion



#write the output to a binary file
#output = output.transpose(2,3,1,0,4).astype(np.int8)


output.tofile("/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/dedispersion/codedisp_output_DM{:03}.bin".format(dm))
print("outputdata size in bytes: ", output.nbytes)
#differemce
print("difference: ", (data.nbytes - output.nbytes)/nant/npol/nchans_per_bridge/ncomplex)





      
    
