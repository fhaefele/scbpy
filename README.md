# SCBpy
Is a Python package containing a collection of functions for audio processing. It is mainly used by the [Sound Communication and Behaviour research group](https://www.sdu.dk/en/forskning/sound-communication-behaviour) at the University of Southern Denmark (SDU).

## Installation
### Direct install from Github
This installs the package directly into your active venv as
```bash
pip install git+ssh://git@github.com:fhaefele/scbpy.git
```
or
```bash
pip install git+https://github.com/fhaefele/scbpy.git
```

### Install it in editable mode
Cloning the repository to your harddrive and install it as editable package via
```bash
git clone https://github.com/fhaefele/scbpy.git
cd ./scbpy
pip install -e .
```
Note: make sure the paths in above are adjusted for your setup. The path in `pip install -e .` refers to the directory in which you just changed with `cd`.

## Usage
After installion you can import the modules/functions directly as
```Python
import scbpy.audio
import scbpy.yin
# or
from scbpy.audio import audioread, audioinfo, spectrum, pwelch
from scbpy.yin import yin
```

For example you can plot the FFT and the PSD like so
```Python
import matplotlib.pyplot as plt
import scbpy.audio

x,fs = scbpy.audio.audioread('/path/to/file.wav')
X,fx = scbpy.audio.spectrum(x,fs)
Pxx, fxx = scbpy.audio.pwelch(x,fs)

plt.figure(1)
plt.clf()
plt.plot(fx / 1e3, X, label="FFT")
plt.plot(fxx / 1e3, Pxx, label="PSD welch")
plt.xlabel("frequency [kHz]")
plt.ylabel("Amplitude [dB/Hz]")
plt.grid()
plt.show()
```

## Scbpy content
Module | Function | Purpose
---|---|---
scbpy.audio | `audioread` | Reading scaled wav-files via scipy
scbpy.audio | `audioinfo` | Audio info from provided path
scbpy.audio | `percent_energy_window` | Returns percent window of input signal
scbpy.audio | `spectrum` | FFT
scbpy.audio | `nextpow2` | Computes the closest power of 2.
scbpy.audio | `pwelch` | PSD
scbpy.audio | `spectrogram` | Spectrogram (scipy>=1.12)
scbpy.audio | `legacy_spectrogram` | Spectrogram (legacy implementation for scipy<1.12)
scbpy.yin | `yin` | Compute the Yin Algorithm
scbpy.yin | `plt_yin` | Plot for Yin
