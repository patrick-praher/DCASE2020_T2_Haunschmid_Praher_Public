from torch.utils.data import Dataset, ConcatDataset
import os
from dcase2020.config import CACHE_DIR
from dcase2020.utils import pickle_load, pickle_dump
import torch
import numpy as np
import librosa

all_devtrain_machines = {
    "dev/ToyCar/train": ['id_01', 'id_02', 'id_03', 'id_04'],
    "dev/ToyConveyor/train": ['id_01', 'id_02', 'id_03'],
    "dev/fan/train": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/pump/train": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/slider/train": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/valve/train": ['id_00', 'id_02', 'id_04', 'id_06']
}


all_devtest_machines = {
    "dev/ToyCar/test": ['id_01', 'id_02', 'id_03', 'id_04'],
    "dev/ToyConveyor/test": ['id_01', 'id_02', 'id_03'],
    "dev/fan/test": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/pump/test": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/slider/test": ['id_00', 'id_02', 'id_04', 'id_06'],
    "dev/valve/test": ['id_00', 'id_02', 'id_04', 'id_06']
}


all_evaltrain_machines = {
    "eval_train/ToyCar/train": ['id_05', 'id_06', 'id_07'],
    "eval_train/ToyConveyor/train": ['id_04', 'id_05', 'id_06'],
    "eval_train/fan/train": ['id_01', 'id_03', 'id_05'],
    "eval_train/pump/train": ['id_01', 'id_03', 'id_05'],
    "eval_train/slider/train": ['id_01', 'id_03', 'id_05'],
    "eval_train/valve/train": ['id_01', 'id_03', 'id_05']
}

all_evaltest_machines = {
    "eval_test/ToyCar/test": ['id_05', 'id_06', 'id_07'],
    "eval_test/ToyConveyor/test": ['id_04', 'id_05', 'id_06'],
    "eval_test/fan/test": ['id_01', 'id_03', 'id_05'],
    "eval_test/pump/test": ['id_01', 'id_03', 'id_05'],
    "eval_test/slider/test": ['id_01', 'id_03', 'id_05'],
    "eval_test/valve/test": ['id_01', 'id_03', 'id_05']
}


machine_id_to_class_label = dict(ToyCar_id_01=0, ToyCar_id_02=1, ToyCar_id_03=2, ToyCar_id_04=3,
                                 ToyConveyor_id_01=4, ToyConveyor_id_02=5, ToyConveyor_id_03=6,
                                 fan_id_00=7, fan_id_02=8, fan_id_04=9, fan_id_06=10,
                                 pump_id_00=11, pump_id_02=12, pump_id_04=13, pump_id_06=14,
                                 slider_id_00=15, slider_id_02=16, slider_id_04=17, slider_id_06=18,
                                 valve_id_00=19, valve_id_02=20, valve_id_04=21, valve_id_06=22, ToyCar_id_05=23, ToyCar_id_06=24, ToyCar_id_07=25,
                                 ToyConveyor_id_04=26, ToyConveyor_id_05=27, ToyConveyor_id_06=28,
                                 fan_id_01=29, fan_id_03=30, fan_id_05=31,
                                 pump_id_01=32, pump_id_03=33, pump_id_05=34,
                                 slider_id_01=35, slider_id_03=36, slider_id_05=37,
                                 valve_id_01=38, valve_id_03=39, valve_id_05=40
                                 )

machine_type_to_class_label = dict(ToyCar=0, ToyConveyor=1, fan=2, pump=3, slider=4, valve=5)


def determine_maximum_snippets(maximum_snippets, frames_per_snippet, preprocessing_params):
    if maximum_snippets == "auto":
        max_frames = librosa.time_to_frames(10, preprocessing_params['sr'],
                                            preprocessing_params['hop_length'],
                                            preprocessing_params['n_fft'])
        maximum_snippets = max_frames // frames_per_snippet
        print("Computed maximum_snippets", maximum_snippets)
    return maximum_snippets


def onehot(y, nr_classes):
    y_tensor = torch.from_numpy(np.asarray([y])).reshape(-1, 1)

    y_onehot = torch.FloatTensor(len(y_tensor), nr_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y_tensor, 1)
    return y_onehot


def prepare_batch(batch, device):
    meta = None
    if len(batch) == 2:
        x, y = batch
    elif len(batch) == 3:
        x, (meta), y = batch
    return x.to(device).float(), meta, y.to(device).float()


def create_unique_dataset_name(subset_name, preprocessing_name, preprocessing_params):
    name = subset_name.replace("/", "_") + "_" + preprocessing_name
    # some runs sorted this in a different way, resulting in duplicates
    sorted_params = ["sr", "n_fft", "hop_length", "power", "n_mels"]
    for key in sorted_params:
        name += "_" + key + str(preprocessing_params[key])
    return name


def get_meta_from_name(file_name):
    basename = os.path.basename(file_name)
    tokens = basename.split("_")
    label = -1
    if tokens[0] != "id":
        assert tokens[0] in ['normal', 'anomaly'], "Unexpected token {}".format(tokens[0])
        label = 0
        if tokens[0] == 'anomaly':
            label = 1
        m_id = tokens[2]
    else:
        m_id = tokens[1]
    return m_id, label


class MachineSoundDataset(Dataset):
    """Machine Sound Dataset. """

    def __init__(self, root_dir, subset_path, machine_id, preprocessing_fn, preprocessing_name,
                 preprocessing_params, use_input_as_target, use_machine_id_as_target,
                 use_machine_type_as_target,
                 maximum_snippets="auto", frames_per_snippet=10,
                 return_meta=False, norm_std_mean=False, mean=None, std=None, nr_classes=None):
        """
        :param root_dir: Path to "DCASE/" dir
        :param preprocessing_fn: function that computes spectrograms
        :param use_input_as_target: used to decided whether to return label
        """
        assert (mean is None and std is None) or (mean is not None and std is not None), "set both mean and std or " \
                                                                                         "none of them"

        assert sum([use_input_as_target, use_machine_id_as_target, use_machine_type_as_target]) <= 1, "at most one of " \
                                                                                                      "them can be " \
                                                                                                      "True "

        self.root_dir = root_dir
        self.subset_path = subset_path
        self.machine_id = machine_id
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_name = preprocessing_name
        self.preprocessing_params = preprocessing_params
        self.use_input_as_target = use_input_as_target
        self.use_machine_id_as_target = use_machine_id_as_target
        self.use_machine_type_as_target = use_machine_type_as_target
        self.frames_per_snippet = frames_per_snippet
        self.maximum_snippets = determine_maximum_snippets(maximum_snippets, frames_per_snippet, preprocessing_params)
        self.return_meta = return_meta
        self.norm_std_mean = norm_std_mean

        self.nr_classes = nr_classes

        self.files = []
        self.labels = []
        for fi in sorted(os.listdir(os.path.join(root_dir, subset_path))):
            if machine_id in fi:
                file_path = os.path.join(root_dir, subset_path, fi)
                m_id, y = get_meta_from_name(file_path)
                self.files.append(file_path)
                self.labels.append(y)

        spectrograms_path = os.path.join(CACHE_DIR, create_unique_dataset_name(subset_path + "_" + str(machine_id), preprocessing_name, preprocessing_params) + ".pt")
        if os.path.exists(spectrograms_path):
            print("Loading data from {} ...".format(spectrograms_path))
            self.spectrograms = pickle_load(spectrograms_path)
        else:
            print("Computing & storing data in {} ...".format(spectrograms_path))
            self.spectrograms = []
            for i, fi in enumerate(self.files):
                self.spectrograms.append(self.preprocessing_fn(os.path.join(self.root_dir, fi), **self.preprocessing_params))
                if i % 100 == 0:
                    print(i)
            pickle_dump(self.spectrograms, spectrograms_path)

        if norm_std_mean and mean is None:
            print("Computing mean and std")
            x = np.zeros((1, preprocessing_params['n_mels']))
            x2 = np.zeros((1, preprocessing_params['n_mels']))
            n = 0
            for spec in self.spectrograms:
                n += spec.shape[-1]
                x += np.sum(spec, axis=-1)
                x2 += np.sum(spec ** 2, axis=-1)
            self.mean = x / n
            self.std = np.sqrt((x2/n) - self.mean **2)
        else:  # will be set to None or to provided mean/std
            self.mean = mean
            self.std = std

        # print("{} normalizing with mean {} and std {}".format(self.subset_path + "_" + self.machine_id,
        #                                                       self.mean[0][:10],
        #                                                       self.std[0][:10]))

    def __len__(self):
        return len(self.files) * self.maximum_snippets

    def __getitem__(self, idx):
        # x = self.preprocessing_fn(os.path.join(self.root_dir, self.files[idx // 30]))
        mapped_idx = idx // self.maximum_snippets
        x = self.spectrograms[mapped_idx]
        start = (idx % self.maximum_snippets) * self.frames_per_snippet
        end =  (idx % self.maximum_snippets) * self.frames_per_snippet + self.frames_per_snippet
        x = x[:,  start:end]
        y = self.labels[mapped_idx]
        if self.norm_std_mean:
            x = ((x.transpose() - self.mean) / self.std).transpose()
        if self.use_input_as_target:
            y = x
        elif self.use_machine_id_as_target:
            class_id = machine_id_to_class_label[self.subset_path.split("/")[1] + "_" + self.machine_id]
            y = onehot(class_id, self.nr_classes)
        elif self.use_machine_type_as_target:
            class_id = machine_type_to_class_label[self.subset_path.split("/")[1]]
            y = onehot(class_id, self.nr_classes)
        if self.return_meta:
            return x, (self.subset_path, self.machine_id, os.path.basename(self.files[mapped_idx])), y
        return x, y


class CombinedMachineSoundDataset(Dataset):
    def __init__(self, root_dir, machines, preprocessing_fn, preprocessing_name,
                 preprocessing_params, use_input_as_target, use_machine_id_as_target,
                 use_machine_type_as_target,
                 maximum_snippets="auto", frames_per_snippet=10, return_meta=False,
                 norm_std_mean=False, mean=None, std=None, norm_per_set=False, nr_classes=None):

        # machines is a dict with machine_type as key and a list of machine_ids as value

        self.set_stats = {}

        if norm_per_set:
            assert norm_std_mean, 'can\'t be false if norm_per_set=True'

        self.maximum_snippets = determine_maximum_snippets(maximum_snippets, frames_per_snippet, preprocessing_params)

        datasets = []
        for m_type in machines.keys():
            machine_ids = machines[m_type]
            for m_id in machine_ids:
                ds = MachineSoundDataset(root_dir, m_type, m_id,
                                         preprocessing_fn=preprocessing_fn,
                                         preprocessing_name=preprocessing_name,
                                         preprocessing_params=preprocessing_params,
                                         use_input_as_target=use_input_as_target,
                                         use_machine_id_as_target=use_machine_id_as_target,
                                         use_machine_type_as_target=use_machine_type_as_target,
                                         maximum_snippets=self.maximum_snippets,
                                         frames_per_snippet=frames_per_snippet,
                                         return_meta=return_meta,
                                         norm_std_mean=norm_per_set,
                                         mean=mean, std=std, nr_classes=nr_classes)
                datasets.append(ds)
                if norm_per_set:
                    self.set_stats[m_type + "_" + m_id] = (ds.mean, ds.std)

        if norm_std_mean and not norm_per_set and mean is None:
            spectrograms = []
            for ds in datasets:
                spectrograms.extend(ds.spectrograms)
            print("Computing mean and std")
            x = np.zeros((1, preprocessing_params['n_mels']))
            x2 = np.zeros((1, preprocessing_params['n_mels']))
            n = 0
            for spec in spectrograms:
                n += spec.shape[-1]
                x += np.sum(spec, axis=-1)
                x2 += np.sum(spec ** 2, axis=-1)
            self.mean = x / n
            self.std = np.sqrt((x2/n) - self.mean **2)

            for ds in datasets:
                ds.mean = self.mean
                ds.std = self.std
                ds.norm_std_mean = True
        else:  # will be set to None or to provided mean/std
            self.mean = mean
            self.std = std

        if not norm_per_set:
            for ds in datasets:
                self.set_stats[ds.subset_path + "_" + ds.machine_id] = (self.mean, self.std)

        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

if __name__ == '__main__':
    from dcase2020.config import DATA_ROOT
    from dcase2020.datasets.preprocessing import baseline_preprocessing
    import librosa.display
    import matplotlib.pyplot as plt

    preprocessing_params = {
        "sr":16000,
        "n_fft":1024,
        "hop_length":512,
        "power":2.0,
        "n_mels": 64
    }

    ds = CombinedMachineSoundDataset(DATA_ROOT,
                                     all_devtrain_machines,
                                     baseline_preprocessing,
                                     "baseline_preprocessing",
                                     preprocessing_params,
                                     use_input_as_target=True,
                                     return_meta=True)

    print("samples:", len(ds))
    for i in [0, 5, 10, len(ds)-1]:
        x, (meta), y = ds[i]
        print(meta)
    print(x.shape)
    plt.figure()
    librosa.display.specshow(x)
    plt.show()
