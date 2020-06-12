import librosa.display
import os
import matplotlib.pyplot as plt
from dcase2020.datasets.preprocessing import baseline_preprocessing
from dcase2020.config import DATA_ROOT
from dcase2020.datasets.machine_sound_dataset import MachineSoundDataset, CombinedMachineSoundDataset

SAMPLE_RATE = 16000
N_FFT = 1024
FRAMES = 5
HOP_LENGTH = 512
POWER = 2.0
N_MELS = 64

selected_samples = ["dev/fan/test/normal_id_02_00000006.wav",
                    "dev/fan/test/anomaly_id_02_00000115.wav"]


for samp in selected_samples:
    log_mel_spectrogram = baseline_preprocessing(file_path=os.path.join(DATA_ROOT, samp),
                                                 sr=SAMPLE_RATE,
                                                 n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH,
                                                 n_mels=N_MELS,
                                                 power=POWER
    )

    print(log_mel_spectrogram.min(), log_mel_spectrogram.max())
    plt.figure()
    librosa.display.specshow(log_mel_spectrogram)
    plt.title(samp)
    plt.show()


preprocessing_params = {
    "sr": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "power": 2.0,
    "n_mels": 128
}

machines = {"dev/fan/train": ['id_00', 'id_06'], "dev/pump/train": ['id_00']}
ds = MachineSoundDataset(DATA_ROOT,
                         "dev/fan/train",
                         "id_00",
                         baseline_preprocessing,
                         "baseline_preprocessing",
                         preprocessing_params,
                         use_input_as_target=True,
                         return_meta=True,
                         norm_std_mean=True)

# for i in range(len(ds)):
#     print(ds[i][0].min(), ds[i][0].max())

spec = ds.spectrograms[0]
plt.figure()
librosa.display.specshow(spec)
plt.savefig("test.png")
spec = ((spec.transpose() - ds.mean) / ds.std).transpose()
plt.figure()
librosa.display.specshow(spec)
plt.savefig("test_norm.png")