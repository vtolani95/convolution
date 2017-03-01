import numpy as np
import scipy.io.wavfile
import time
import pyaudio

# roll array x by k elements
# zero out the last k elements of x
def roll_zero(x, k):
  result = x[k:]
  result = np.append(x[k:], np.zeros(k))
  return result

def zero_pad(x, k):
  return np.append(x, np.zeros(k))

# partition impulse response and precompute frequency response for each block
def precompute_frequency_responses(h, L, k, num_blocks):
  H = np.zeros((num_blocks, L+k)).astype('complex128')
  for j in range(num_blocks):
    H[j] += np.fft.fft(zero_pad(h[j*k: (j+1)*k], L))
  return H

# FFT Based Conv
def fft_conv(x, h):
  L, P = len(x), len(h)
  h_zp = zero_pad(h, L-1)
  x_zp = zero_pad(x, P-1)
  X = np.fft.fft(x_zp)
  start_time = time.time()
  output = np.fft.ifft(X * np.fft.fft(h_zp)).real
  output = amount_verb * output + x_zp
  end_time = time.time()
  return output, end_time - start_time

# convolve entire x with h via overlap add
def overlap_add_conv(x, h):
  k = 2 ** 12
  L, P = len(x), len(h)
  num_blocks = int(P / k)
  output = np.zeros(L+P-1)
  X = np.fft.fft(zero_pad(x, k-1))
  start_time = time.time()
  for i in range(num_blocks):
    ir_block = h[i*k:(i+1)*k]
    fr_block = np.fft.fft(np.append(ir_block, np.zeros(L-1)))
    output[i*k:(i+1)*k+L-1] += np.fft.ifft(fr_block * X).real
  x_zp = zero_pad(x, P-1)
  output = amount_verb * output + x_zp
  return output, time.time() - start_time

# break sig into chunks (size L)
# break ir into chunks (size k)
# use overlap add fft based conv to convolve
def uniform_partitioned_conv(x, h):
  P = len(h)
  L = 2**8 #signal block size
  k = L #ir block size

  x = x[0: -1 * (x.shape[0] % L)]
  N = x.shape[0]
  num_ir_blocks = int(P/k)
  num_sig_blocks = int(x.shape[0] / L)
  H = precompute_frequency_responses(h, L, k, num_ir_blocks)
  output = np.zeros(P-1 + num_sig_blocks*L)
  start_time = time.time()
  for i in range(num_sig_blocks):
    input_buffer = zero_pad(x[i*L: (i+1)*L], k)
    spectrum = np.fft.fft(input_buffer)
    for j in range(num_ir_blocks):
      output[i*L+j*k: (i+1)*L+(j+1)*k-1] += np.fft.ifft(spectrum * H[j]).real[:2*k-1]
  x_zp = zero_pad(x, P-1)
  output = amount_verb * output + x_zp
  return output, time.time() - start_time


# Frequency Domain Delay Line
# Do Overlap Add in Frequency Domain
def fdl(x, h):
  L = 2**8 #signal block size
  p = len(h)
  k = L # ir block size

  num_ir_blocks = int(p/k)
  num_sig_blocks = int(len(x) / L)
  H = precompute_frequency_responses(h, L, k, num_ir_blocks)
  fdl = np.zeros(2*L*num_ir_blocks).astype('complex128')
  output = np.zeros(p+len(x)-1).astype('float64')
  out = np.zeros(2*L-1)

  start_time = time.time()
  for i in range(num_sig_blocks):
    input_buffer = x[i*L: (i+1)*L]
    spectrum = np.fft.fft(zero_pad(input_buffer, L))
    for j in range(num_ir_blocks):
      fdl[j*2*L: (j+1)*2*L] += H[j] * spectrum
    out += np.fft.ifft(fdl[:2*L]).real[:2*L-1]
    output[i*L:(i+1)*L] += out[:L]
    fdl = roll_zero(fdl, 2*L)
    out = roll_zero(out, L)
  for i in range(1, num_ir_blocks): #process remaining frequency blocks
    out += np.fft.ifft(fdl[:2*L]).real[:2*L-1]
    output[num_sig_blocks+i*L: num_sig_blocks+(i+1)*L] += out[:L]
    out = roll_zero(out, L)
    fdl = roll_zero(fdl, 2*L)
  x_zp = zero_pad(x, p-1)
  output = amount_verb  * output + x_zp
  return output, time.time() - start_time


### Script
### All convolution Algorithmns for One Channel Real Signals
fs1, guitar = scipy.io.wavfile.read('./sounds/blues_guitar.wav')
fs2, reverb = scipy.io.wavfile.read('./sounds/PlateSmall_01.wav')
reverb = (reverb.astype('float64') / np.max(reverb))[:,0] #one channel only
amount_verb = .015

sig, timer = fft_conv(guitar, reverb)
scipy.io.wavfile.write('./processed/fft_conv.wav', fs1, sig.astype('int16'))
print('Naive FFT Conv Reverb: ' + str(timer) + ' seconds')

sig2, timer = overlap_add_conv(guitar, reverb)
scipy.io.wavfile.write('./processed/overlap_add_conv.wav', fs1, sig2.astype('int16'))
print('Overlap Add Conv Reverb: ' + str(timer) + ' seconds')

sig3, timer = uniform_partitioned_conv(guitar, reverb)
scipy.io.wavfile.write('./processed/uniform_partitioned_conv.wav', fs1, sig3.astype('int16'))
print('Uniform Partitioned Conv Reverb: ' + str(timer) + ' seconds')

sig4, timer = fdl(guitar, reverb)
scipy.io.wavfile.write('./processed/fdl_conv.wav', fs1, sig4.astype('int16'))
print("FDL: " + str(timer) + ' seconds')






