import librosa
import numpy as np
import sys


def baseline_preprocessing(file_path, sr, n_fft, hop_length, n_mels, power):
    # Preprocessing as performed for baseline system:
    # https://github.com/y-kawagu/dcase2020_task2_baseline/blob/master/common.py#L119
    y, _ = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram
