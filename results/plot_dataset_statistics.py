from dcase2020.datasets.machine_sound_dataset import all_devtrain_machines, MachineSoundDataset
from dcase2020.config import DATA_ROOT
from dcase2020.datasets.preprocessing import baseline_preprocessing
import numpy as np
import matplotlib.pyplot as plt
import os

preprocessing_params = {
    "sr": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "power": 2.0,
    "n_mels": 128
}

n_datasets = sum([len(m_ids) for _, m_ids in all_devtrain_machines.items()])
ds_means = np.zeros((preprocessing_params['n_mels'], n_datasets))
ds_stds = np.zeros((preprocessing_params['n_mels'], n_datasets))
ds_lengths = []
ds_names = []

datasets = []
i = 0
for m_type in all_devtrain_machines.keys():
    for m_id in all_devtrain_machines[m_type]:
        ds = MachineSoundDataset(
            DATA_ROOT, m_type, m_id,
            preprocessing_fn=baseline_preprocessing,
            preprocessing_name="baseline_preprocessing",
            preprocessing_params=preprocessing_params,
            use_input_as_target=False,
            use_machine_id_as_target=False,
            use_machine_type_as_target=False,
            maximum_snippets="auto",
            frames_per_snippet=8,
            # return_meta=True,
            norm_std_mean=True#,
            # mean=mean, std=std
        )
        ds_means[:, i] = ds.mean
        ds_stds[:, i] = ds.std
        ds_names.append(m_type + "_" + m_id)
        ds_lengths.append(len(ds.files))
        datasets.append(ds)
        i += 1


## heatmap
# plt.imshow(ds_means)
# plt.colorbar()
# plt.show()


# nr. of audios
x = range(len(ds_lengths))
plt.bar(x, ds_lengths)
plt.xticks(ticks=x, labels=ds_names, rotation=90)
plt.show()

# lines
plt.figure(figsize=(15, 10))
plt.plot(ds_means)
plt.legend(ds_names)
plt.savefig(os.path.expanduser("~/Repos/DCASE_2020/visualisations/dataset_means.png"))
plt.xlabel("Frequency bins")
plt.show()