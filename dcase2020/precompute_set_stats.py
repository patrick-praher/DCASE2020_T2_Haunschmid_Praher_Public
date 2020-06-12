from dcase2020.datasets.machine_sound_dataset import CombinedMachineSoundDataset, all_devtrain_machines, \
    all_evaltrain_machines
from dcase2020.datasets.preprocessing import baseline_preprocessing
from dcase2020.config import DATA_ROOT
from dcase2020.utils import pickle_dump
from train_anomaly_detector import train_config


def compute_set_stats(norm_per_set):
    _config = train_config()

    use_input_as_target = False
    use_machine_type_as_target = False
    use_machine_id_as_target = False
    cond_label_size = None

    train_machines = all_devtrain_machines
    train_machines.update(all_evaltrain_machines)

    devset = CombinedMachineSoundDataset(DATA_ROOT,
                                         train_machines,
                                         preprocessing_fn=baseline_preprocessing,
                                         preprocessing_name="baseline_preprocessing",
                                         preprocessing_params=_config['preprocessing_params'],
                                         use_input_as_target=use_input_as_target,
                                         use_machine_type_as_target=use_machine_type_as_target,
                                         use_machine_id_as_target=use_machine_id_as_target,
                                         maximum_snippets=_config['maximum_snippets'],
                                         frames_per_snippet=_config['frames_per_snippet'],
                                         return_meta=True,
                                         norm_std_mean=_config["apply_normalization"],
                                         norm_per_set=norm_per_set,
                                         nr_classes=cond_label_size)

    return devset.set_stats


if __name__ == '__main__':

    set_stats0 = compute_set_stats(False)
    pickle_dump(set_stats0, 'stats/allstats_nps0.pt')

    set_stats1 = compute_set_stats(True)
    pickle_dump(set_stats1, 'stats/allstats_nps1.pt')