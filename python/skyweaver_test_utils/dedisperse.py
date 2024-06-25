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
nsamples = 32768
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
max_delay = 8192

print("central_f0: ", cfreq)
print("bridge_bw: ", bridge_bw)
print("tsamp: ", tsamp)
print("dm: ", dm)
print("max_delay: ", max_delay)
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

fftlen = max_delay

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

chunks = int(nsamples/fftlen)

# create a random array of data
data = np.random.randint(0, 2**nbits, (nant, npol, nchans_per_bridge, nsamples, ncomplex))


data.tofile("/tmp/codedisp_input.dat")

truncated_length = fftlen - max_shift

output = np.zeros((nant, npol, nchans_per_bridge, chunks*truncated_length), dtype='complex64')


print("Starting dedispersion....")

offset = int(max_shift/2) if max_shift%2 == 0 else int(max_shift/2.0) + 1
first = True

for ant in range(nant):
   for pol in range(npol):
      first = True
      for i in range(chunks):
         d = data[ant,  pol, :,  0:fftlen,:] if first else data[ant, pol, :, i*truncated_length: i*truncated_length + fftlen, :]
         d = d[:,:,0] + 1j*d[:,:,1]
         dedispersed_data = np.fft.ifft(np.fft.fft(d.astype(np.complex64), axis=1) * chirps, axis=1)
         output[ant, pol, :,  i*truncated_length:(i+1)*truncated_length] = dedispersed_data[:,int(max_shift/2.0): fftlen - offset]    
         first = False



#write the output to a binary file
output.tofile("/tmp/codedisp_output.dat")





      
    
