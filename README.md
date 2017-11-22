# Realtime Convolution Algorithms for Reverb Signals
This code base deals with several algorithms for rapid convolution of two sequences. I explore how to optimize for near real time processing of signals with the ultimate goal of applying "reverb" to a input sound on a single core operating system. These algorithms are prototyped in python for readability, but will ultimately be deployed in C. The `F` operator represents a DFT and `d` is a delta function.

**All algorithmns input x and h and output y:**
x- input signal
    h- filter
    y- input with filter applied    
`y[n] = x[n] * h[n]`  

[**fft_conv**](https://github.com/vtolani95/convolution_algorithms/blob/master/reverb.py#L24)  
Traditional Convolution is a O(n^2) algorithm. Fourier Transform properties tell us that a convolution in time corresponds to multiplication in frequency. Using the FFT algorithm (O(nlogn)) for efficient transformation to frequency domain we can transform both signals to frequency, multiply, and transform back giving a O(nlogn) algorithm for convolution. Note- for accurate linear convolution we must zero pad the signals appropriately to avoid aliasing.  
`y[n] = F^-1(F(x[n]) * F(h[n]))`  

[**overlap_add_conv**](https://github.com/vtolani95/convolution_algorithms/blob/master/reverb.py#L36)  
Convolution is a linear operator so we can break up the impulse response into chunks of size k as follows:  
`h[n] = [h_1[n]|h_2[n]|h_3[n]|...]`  
`h[n] = h_1[n] + h_2[n]*d[k] + h_3[n]*d[2k] + ...`  
Convolution is a linear operator so we have:  
`y[n] = x[n] * h[n]`  
`y[n] = x[n] * (h_1[n] + h_2[n]*d[k] + h_3[n]*d[2k] + ...)`  
`y[n] = x[n]*h_1[n] + x[n]*h_2[n]*d[k] + x[n]*h_3[n]*d[2k] + ...)`  
Y can be expressed as a sum of convolved with each block of h shifted according to its position in h. Each of these "mini convolutions" can be done via the fft method mentioned above.

[**uniform_partitioned_conv**](https://github.com/vtolani95/convolution_algorithms/blob/master/reverb.py#L54)  
Now we imagine that x[n] is a realtime signal (audio perhaps) that we process in input buffer chunks. We aim to read input a buffer, process, and output a buffer in near realtime (under ~8ms is acceptable for audio applications). Each buffer length L is read in, convolved with h[n] (via overlap add as described above) resulting in a sequence L+P-1 long. We then output the first L points to an output buffer. The remaining P-1 are stored in an output "delay line". These points are added with the first L points that result from convolving the **second** buffer with h[n]. In this sense we have an outer overlap add scheme that divides x into input buffers and convolves each with h[n] via another overlap add. Here I break a prerecorded signal into chunks to simulate an input buffer.

[**fdl- Frequency Domain Delay Line**](https://github.com/vtolani95/convolution_algorithms/blob/master/reverb.py#L78)  
Again we have x[n] being a realtime signal that we process in chunks. We still aim to read in input buffer, process, and output a buffer in near realtime. Now however we aim to perform fewer fft operations. Following the method described in [section 2.1](http://ericbattenberg.com/school/partconvDAFx2011.pdf) if we choose the impluse response block size, k, to be k=L (L is the input buffer size) we can actually perform overlap add in the frequency domain and only use one output fft per input cycle. This can lead to substantial speed ups in realtime audio processing. Initial tests in python show substantial speed ups when compared to uniform partitioned convolution.

**Future Work**  
1. Optimize fft size (fft works best when the input sequence size is a power of 2)  
2. Add non uniform partitions to impulse response as described in [section 2.2](http://ericbattenberg.com/school/partconvDAFx2011.pdf). The earlier impulse responses will be short for low latency while the later ones will be long for computational efficiency.  
3. Use Decimation in Frequency to construct a time distributed approach for scheduling necessary FFT computations as described in [section 4](http://ericbattenberg.com/school/partconvDAFx2011.pdf).  
4. Reduce FFT computation by using conjugate symmetry induced by entirely real audio signals. Method described [here](http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM) 
5. Port code to C for near real time processing
