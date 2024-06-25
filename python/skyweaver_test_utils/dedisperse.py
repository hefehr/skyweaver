import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc


def dm_delay(nu1, nu2, dm, tsamp):
   return ((1/pow(nu1/1000,2.0)) - (1/pow(nu2/1000, 2.0)))*dm*0.00415/tsamp

# create TAFTP voltages

nant =  57
npol = 2
ncomplex = 2
nsamples = 1024
nbits = 8
dm  = 100

cfreq = 1.284e3 #MHz
total_bw = 856 # MHz
total_nchans = 4096
coarse_chan_width = total_bw / total_nchans


nbridges = 64
nchans_per_bridge = int(total_nchans / nbridges)
bridge_bw = total_bw / nbridges

ibridge = 0

tsamp = 4096/856e6
fftlen = 256

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


chirps = np.zeros((nchans_per_bridge, fftlen),dtype='complex64')
chirp_chan_width = coarse_chan_width/fftlen
print("chirp_chan_width: ", chirp_chan_width)

def chirp(flow, df, dm):
   s = (-2*np.pi*dm*148806423e9)
   phase = (df * df * s)/((flow + df) * flow * flow)
   return (np.exp(1j*phase))


fine_chans = np.linspace(0, coarse_chan_width, fftlen)
print("fine_chans: ", fine_chans)

for i in range(nchans_per_bridge):
   chirps[i] = chirp(chanedges[i], fine_chans, dm)



print("computing the max shift from the dm...")
#generate max shifts for all channels
max_shift = int(dm_delay(chanedges[0], chanedges[1],dm, tsamp))
print("max_shift: ", max_shift)
max_shift = max_shift + 1 if max_shift%2 != 0 else max_shift



# create a random array of data
data = np.random.randint(0, 2**nbits, (nant, npol, nchans_per_bridge, nsamples, ncomplex)).astype(np.int8)

#print size in bytes
print("data size in bytes: ", data.nbytes)
print("data shape: ", data.shape)

data.tofile("/tmp/codedisp_input.dat")

truncated_length = fftlen - max_shift

chunks = int(nsamples/truncated_length)

print("chunks: ", chunks)

print(truncated_length, fftlen, max_shift)
offset = int(max_shift/2) 


output = np.zeros((nant, npol, nchans_per_bridge, nsamples - max_shift, ncomplex), dtype = np.int8)
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
         d = data[ant,  pol, :, start : start + fftlen,:] 
         print("start: ", start, " end: ", start + fftlen, "d.shape: ", d.shape)

         if(d.shape[1] != fftlen):
            print("Padding the data:",fftlen-d.shape[1])
            temp = np.zeros((nchans_per_bridge, fftlen-d.shape[1], ncomplex), dtype = np.int8)
            d = np.concatenate((d, temp), axis=1)
            print("d.shape: ", d.shape)

         d = d[:,:,0] + 1j*d[:,:,1]
         dedispersed_data = np.fft.ifft(np.fft.fft(d.astype(np.complex64), axis=1) * chirps, axis=1)

              
         
         real = dedispersed_data[:,offset : fftlen - offset].real.astype(np.int8)
         imag = dedispersed_data[:,offset : fftlen - offset].imag.astype(np.int8)

         if((i+1)*truncated_length > (nsamples - max_shift)):
            print("last chunk")
            output[ant, pol, :,  i*truncated_length:(nsamples - max_shift), 0] = real[:,  0:(nsamples - max_shift-i*truncated_length)]
            output[ant, pol, :,  i*truncated_length:(nsamples - max_shift), 1] =  imag[:,  0:(nsamples - max_shift-i*truncated_length)]
         else:
            output[ant, pol, :,  i*truncated_length:(i+1)*truncated_length, 0] = real 
            output[ant, pol, :,  i*truncated_length:(i+1)*truncated_length, 1] =  imag
         first = False
      break
   break







#write the output to a binary file
output.tofile("/tmp/codedisp_output.dat")
print("outputdata size in bytes: ", output.nbytes)
#differemce
print("difference: ", (data.nbytes - output.nbytes)/nant/npol/nchans_per_bridge/ncomplex)





      
    
