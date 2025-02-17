import warnings
import wave
from collections import namedtuple

import numpy as np
import scipy.fft
import scipy.io
import scipy.signal


def audioread(file_path: str) -> tuple[np.ndarray, int]:
    """
    Reads an audio file in WAV format and normalizes its amplitude to +/- 1
    using scipy.io.wavfile module.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file.

    Returns:
    --------
    tuple[np.ndarray, int]
        - x : np.ndarray
            The audio signal as a NumPy array. If the wav is recorded in 16-bit or 32-bit,
            it is normalized to the range [-1, 1]. If the input is already float (32-bit
            or 64-bit), it is returned as is. For more information check scipy.io.wavfile.read.
        - fs : int
            Sampling rate of the audio file.

    Raises:
    -------
    ValueError
        If the audio format is unsupported and cannot be properly scaled.
    """
    fs, x = scipy.io.wavfile.read(file_path)
    if x.dtype == "int16":
        nb_bits = 16
    elif x.dtype == "int32":
        nb_bits = 32
    elif x.dtype == "float32" or x.dtype == "float64":
        return x, fs
    else:
        raise ValueError("Can't scale audio file properly")
    max_nb_bit = float(2 ** (nb_bits - 1))
    x = x / max_nb_bit
    return x, fs


def audioinfo(file_path: str) -> namedtuple:
    """
    Extracts and returns metadata about an audio file in WAV format.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file.

    Returns:
    --------
    Audioinfo : namedtuple
        A named tuple containing the following audio metadata:
        - sample_rate (int): The sample rate of the audio file in Hz.
        - num_channels (int): The number of audio channels.
        - total_samples (int): The total number of audio samples in the file.
        - duration_s (float): The duration of the audio file in seconds.
        - bits_per_sample (int): The bit depth of the audio file (e.g., 16, 24, 32).

    Raises:
    -------
    wave.Error
        If the file is not a valid WAV file.
    """
    Audioinfo = namedtuple(
        "Audioinfo",
        [
            "sample_rate",
            "num_channels",
            "total_samples",
            "duration_s",
            "bits_per_sample",
        ],
    )
    file_path = str(file_path)  # in case its a Path object
    with wave.open(file_path, "rb") as w:
        a = w.getparams()
    return Audioinfo(
        a.framerate,
        a.nchannels,
        a.nframes,
        float(a.nframes) / float(a.framerate),
        a.sampwidth * 8,
    )


def percent_energy_window(x: np.ndarray, percent: float = 0.95) -> np.ndarray:
    """
    Computes a boolean mask that identifies the central window of a signal containing a
    specified percentage of its total energy.

    Parameters:
    -----------
    x : np.ndarray
        The input signal as a NumPy array.
    percent : float, optional (default=0.95)
        The fraction of the total signal energy to be retained within the window.
        Must be in the range (0,1).

    Returns:
    --------
    np.ndarray
        A boolean array of the same length as `x`, where `True` indicates the indices
        that fall within the central energy window.

    Raises:
    -------
    ValueError
        If `percent` is not within the range [0,1].

    Notes:
    ------
    - The function calculates the cumulative sum of the squared signal (`x**2`), normalizes it
      by its maximum value, and then finds indices corresponding to the given energy percentage.
    - The lower and upper limits define the symmetric range around the central energy region.
    """
    if not (0 <= percent <= 1):
        raise ValueError("percent must be in the range [0,1]")
    lower_lim = (1 - percent) / 2
    upper_lim = 1 - lower_lim
    csig = np.cumsum(x**2)
    csig = csig / np.max(csig)
    return (csig > lower_lim) & (csig < upper_lim)


def spectrum(x, fs, nfft=None, p0=1):
    """
    Computes the single-sided amplitude spectrum of a signal in decibels (dB).

    Parameters:
    -----------
    x : np.ndarray
        The input signal.
    fs : float
        The sampling frequency of the signal in Hz.
    nfft : int, optional
        The number of points for the FFT computation. If None, it defaults to the length of `x`.
        Must be greater than or equal to `x.shape[0]`, otherwise an error is raised.
    p0 : float, optional (default=1)
        Reference amplitude for computing the spectrum in decibels.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        - X (np.ndarray): The magnitude spectrum in dB.
        - fx (np.ndarray): The corresponding frequency values in Hz.

    Raises:
    -------
    ValueError
        If `nfft` is smaller than `x.shape[0]`, as this would truncate the signal.

    Notes:
    ------
    - The function computes the Fast Fourier Transform (FFT) of the input signal and extracts
      the positive frequency components (single-sided spectrum).
    - The output spectrum is scaled to account for energy distribution in the FFT.
    - The result is expressed in decibels relative to `p0`.
    - The term `- 10*np.log10(fs/nfft)` normalizes the spectral density.
    """
    if nfft is None:
        nfft = x.shape[0]
    if nfft < x.shape[0]:
        raise ValueError(
            "nfft must be greater than or equal to the length of x to prevent truncation."
        )
    X = scipy.fft.fft(x, nfft)[: nfft // 2]
    fx = scipy.fft.fftfreq(nfft, 1 / fs)[: nfft // 2]
    X = 20 * np.log10(2 / nfft * np.abs(X) / p0) - 10 * np.log10(fs / nfft)
    return X, fx


def nextpow2(x: int | float) -> int:
    """
    Computes the next power of 2 exponent that is greater than or equal to the given number.

    Parameters:
    -----------
    x : int | float
        The input number.

    Returns:
    --------
    int
        The smallest integer `p` such that `2**p` is greater than or equal to `x`.

    Raises:
    -------
    ValueError
        If `x` is not positive, since logarithm is undefined for non-positive values.
    """
    if x <= 0:
        raise ValueError("Input must be a positive number.")
    return np.ceil(np.log2(x))


def compute_default_window(sig_len: int, window_fct: int | np.ndarray | None = None) -> np.ndarray:
    """
    Computes a default window function based on the signal length.

    Parameters:
    -----------
    sig_len : int
        The length of the input signal.
    window_fct : int, np.ndarray, or None, optional
        - If None, the function automatically selects an appropriate window length.
        - If an integer, it specifies the window length, and a Hamming window is applied.
        - If an array, it is used as the window function directly.

    Returns:
    --------
    np.ndarray
        The computed window function.

    Notes:
    ------
    - Default window size selection based on `x.shape[0]`, adapting for different signal lengths.
    - If `window_fct` is an integer, a Hamming window of that size is used.
    - If an array, it is used as the window function directly.
    """
    if window_fct is None:
        if sig_len < 2**6:
            d = 2
        elif sig_len < 2**11:
            d = 8
        elif sig_len < 2**12:
            d = 16
        elif sig_len < 2**13:
            d = 32
        elif sig_len < 2**14:
            d = 64
        else:
            d = 128
        window_fct = int(sig_len / d)
    if np.isscalar(window_fct):
        window_fct = scipy.signal.windows.hamming(int(window_fct))
    return window_fct


def pwelch(
    x: np.ndarray,
    fs: int,
    window_fct: int | np.ndarray | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Power Spectral Density (PSD) estimate using Welch's method.

    This function is a wrapper around `scipy.signal.welch`, providing automatic window size
    selection based on the input signal length.

    Parameters:
    -----------
    x : np.ndarray
        Input signal.
    fs : int
        Sampling frequency of the signal in Hz.
    window_fct : int, np.ndarray, or None, optional (default=None)
        Window will be computed by scbpy.audio.compute_default_window().
    overlap : float, optional (default=0.5)
        Fraction of window overlap (0 to 1).
    nfft : int or None, optional (default=None)
        Number of FFT points. If None, it is set to the next power of 2 of `x.shape[0]`,
        with a maximum limit of `2**16` (65536).

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        - X (np.ndarray): Power Spectral Density (PSD) in dB/Hz.
        - fx (np.ndarray): Frequency values corresponding to the PSD estimate.

    Notes:
    ------
    - Default window size selection based on `x.shape[0]`, adapting for different signal lengths.
    - If `window_fct` is an integer, a Hamming window of that size is used.
    - The output PSD is converted to decibels (dB/Hz).
    """
    window_fct = compute_default_window(x.shape[0], window_fct=window_fct)
    if nfft is None:
        nfft = int(2 ** nextpow2(x.shape[0]))
        if nfft > 2**16:  # limit default to max fft size of 65536
            nfft = 2**16
    if nfft < window_fct.shape[0]:
        raise ValueError(
            f"nfft must be greater than or equal to the window length {window_fct.shape[0]} "
            "to prevent truncation."
        )
    fx, X = scipy.signal.welch(
        x,
        fs,
        window=window_fct,
        noverlap=int(overlap * window_fct.shape[0]),
        nfft=nfft,
        detrend=False,
        return_onesided=True,
    )
    X = 10 * np.log10(X)  # scale for dB/Hz
    return X, fx


def legacy_spectrogram(
    x: np.ndarray,
    fs: int,
    window_fct: int | np.ndarray | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the spectrogram of a signal using the Short-Time Fourier Transform (STFT).

    Parameters:
    -----------
    x : np.ndarray
        Input signal.
    fs : int
        Sampling frequency of the signal in Hz.
    window_fct : int, np.ndarray, or None, optional
        Window will be computed by scbpy.audio.compute_default_window().
    overlap : float, optional (default=0.5)
        Fraction of window overlap (0 to 1).
    nfft : int or None, optional
        Number of FFT points. If None, it is set to the next power of 2 of `x.shape[0]`,
        with a maximum limit of `2**16` (65536).

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - f (np.ndarray): Array of frequency bins in Hz.
        - t (np.ndarray): Array of segment time indices in seconds.
        - Sxx (np.ndarray): Spectrogram of the signal, representing the power spectral density.

    Raises:
    -------
    ValueError
        If `overlap` is not within the range [0,1].

    Notes:
    ------
    - This function uses the legacy implementation of `scipy.signal.spectrogram`.
    - The spectrogram is computed using a windowed STFT with a specified overlap.
    - The power spectral density is returned in the `scaling="spectrum"` mode.
    """
    warnings.warn(
        "Uses legacy implementation of scipy.signal.spectrogram.", DeprecationWarning, stacklevel=2
    )
    if window_fct is None:
        window_fct = compute_default_window(x.shape[0], window_fct=window_fct)
    if nfft is None:
        nfft = int(2 ** nextpow2(x.shape[0]))
        if nfft > 2**16:  # limit default to max fft size of 65536
            nfft = 2**16
    if nfft < window_fct.shape[0]:
        raise ValueError(
            f"nfft must be greater than or equal to the window length {window_fct.shape[0]} "
            "to prevent truncation."
        )

    f, t, Sxx = scipy.signal.spectrogram(
        x,
        fs,
        window=window_fct,
        noverlap=int(overlap * window_fct.shape[0]),
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
    )
    return f, t, Sxx


def spectorgram(
    x: np.ndarray,
    fs: int,
    window_fct: int | np.ndarray | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, scipy.signal.ShortTimeFFT]:
    """
    Computes the spectrogram of a signal using the Short-Time Fourier Transform (STFT).

    Parameters:
    -----------
    x : np.ndarray
        Input signal.
    fs : int
        Sampling frequency of the signal in Hz.
    window_fct : int, np.ndarray, or None, optional
        Window will be computed by scbpy.audio.compute_default_window().
    overlap : float, optional (default=0.5)
        Fraction of window overlap (0 to 1).
    nfft : int or None, optional
        Number of FFT points. If None, it is set to the next power of 2 of `x.shape[0]`,
        with a maximum limit of `2**16` (65536).

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - f (np.ndarray): Array of frequency bins in Hz.
        - t (np.ndarray): Array of segment time indices in seconds.
        - Sxx (np.ndarray): Spectrogram of the signal, representing the power spectral density.
        - SFT (scipy.signal.ShortTimeFFT): The Short-Time Fourier Transform object.

    Raises:
    -------
    ValueError
        If `overlap` is not within the range [0,1].

    Notes:
    ------
    - Uses scipy.signal.ShorTimeFFT class to compute the spectrogram.
    - The spectrogram is computed using a windowed STFT with a specified overlap.
    - The power spectral density is returned in the `scale_to="magnitude"` mode.
    """
    if window_fct is None:
        window_fct = compute_default_window(x.shape[0], window_fct=window_fct)
    if nfft is None:
        nfft = int(2 ** nextpow2(x.shape[0]))
        if nfft > 2**16:  # limit default to max fft size of 65536
            nfft = 2**16
    if nfft < window_fct.shape[0]:
        raise ValueError(
            f"nfft must be greater than or equal to the window length {window_fct.shape[0]} "
            "to prevent truncation."
        )
    SFT = scipy.signal.ShortTimeFFT(
        window_fct,
        hop=int((1 - overlap) * window_fct.shape[0]),
        fs=fs,
        mfft=nfft,
        scale_to="magnitude",
    )
    return SFT.f, SFT.t(x.shape[0]), SFT.spectrogram(x, detr="constant"), SFT
