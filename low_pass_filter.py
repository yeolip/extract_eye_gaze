# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
from matplotlib import pyplot as plt

# Filter requirements.
order = 3
fs = 20.0  # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

''' 
fs - sample rate, Hz, 
cutoff - desired cutoff frequency of the filter, Hz
order 
'''
def set_param_low_pass_filter(sample_rate, cutoff_freq, order_s):
    global order, fs, cutoff
    order =  order_s
    fs = sample_rate
    cutoff = cutoff_freq
    pass

def low_pass_filter(data, window, padding=1):
    if(len(data)<window):
        print('data size should be longer than window size!!', len(data), window)

    out = []
    buf = []
    # print(data)
    for i in range(0,len(data),window):
        # print(i, data[i:i+twindow])
        # pad = np.zeros(padding)
        pad = [0] * padding
        # print(pad)
        buf.extend(pad)
        # print(buf)
        # buf.extend(data[i:min(i+window,len(data)-i)])
        buf.extend(data[i:i+window])
        buf.extend(pad)
        # print(buf)
        ty = butter_lowpass_filter(buf, cutoff, fs, order)
        out.extend(ty[padding:padding+window])
        # print("output_t", ty[padding:padding+window])
        # print("output_t2", ty)

        # print("in-put", data[i:i+twindow])
        # print("output", ty)
    # print('final out', out)
    return out

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def test_use_previous_to_one_result(data, window):
    y = []
    for i in range(0, len(data)-window, 1):
        # print(i, 'data=',data[i:i + window],'windows=',window)
        ty = low_pass_filter(data[i:i + window], window, padding=0)
        # print('ty',ty)
        y.extend([ty[-1]])
        # y.extend([ty[int(window/2)]])

    tyy = [0] * window
    tyy.extend(y)
    return tyy

def test_use_previous_after_to_one_result(data, window):
    y = []
    for i in range(0, len(data)-window, 1):
        # print(i, 'data=',data[i:i + window],'windows=',window)
        ty = low_pass_filter(data[i:i + window], window, padding=1)
        # print('ty',ty)
        # y.extend([ty[-1]])
        y.extend([ty[int(window/2)]])

    tyy = [0] * int(window/2)
    tyy.extend(y)
    tyy.extend([0] * int(window/2))
    if(len(data)-len(tyy) != 0):
        tyy.extend([0] * (len(data)-len(tyy)))
    return tyy

if __name__ == '__main__':
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 1.0             # seconds
    n = int(T * fs)     # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) \
            + 0.5*np.sin(12.0*2*np.pi*t)
    print('data',data.shape)

    #앞뒤로 커널사이즈를 0 넣어주고 필터링해보자
    # Filter the data, and plot both the original and filtered signals.
    # y = butter_lowpass_filter(data, cutoff, fs, order)
    twindow = 15
    # y = []
    # # y = [0] * len(data)
    # for i in range(0, len(data)-twindow, 1):
    #     print(i, 'data=',data[i:i + twindow],'windows=',twindow)
    #     ty = low_pass_filter(data[i:i + twindow], twindow, padding=1)
    #     print('ty',ty)
    #     # y.extend([ty[-1]])
    #     y.extend([ty[int(twindow/2)]])

    # print(len(y), len(data[twindow-1:-1]))
    # y = low_pass_filter(data, twindow, padding=1)
    # y = low_pass_filter2(data, 20)

    # testtt = [[340.0], [323.0], [323.5], [322.5], [321.5], [318.0], [315.0], [312.0], [308.0], [308.5] [318.0], [315.0], [312.0], [308.0], [308.5]] * 1
    # print(data.shape, data)
    # testtt = [340.0, 323.0, 323.5, 322.5, 321.5, 318.0, 315.0, 312.0, 308.0, 308.5, 318.0, 315.0, 312.0, 308.0, 308.5] * 2
    # testtt = np.array([280.,  279.,  277.5, 276.5, 275.,  276.5, 279.5, 283.5, 287.,  288.,  287.5, 288.,289.5, 288.5, 288.])
    y = test_use_previous_after_to_one_result(data, twindow)
    print('y', y)
    # y = test_use_previous_to_one_result(data, twindow)

    # buf = []
    # # print(data)
    # for i in range(0,int(fs*T),twindow):
    #     # print(i, data[i:i+twindow])
    #     buf = [0]
    #     buf.extend(data[i:i+twindow])
    #     buf.extend([0])
    #     print(buf)
    #     # ty = butter_lowpass_filter(data[i:i+twindow], cutoff, fs, order)
    #     ty = butter_lowpass_filter(buf, cutoff, fs, order)
    #     y.extend(ty[1:1+twindow])
    #     print("output_t", ty[1:1+twindow])
    #     # print("output_t2", ty)
    #
    #     # print("in-put", data[i:i+twindow])
    #     # print("output", ty)

    # tyy = [0] * twindow
    # tyy.extend(y)
    # tyy = [0] * int(twindow/2)
    # tyy.extend(y)
    # tyy.extend([0] * int(twindow/2))


    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()