from dcase2020.datasets.machine_sound_dataset import all_devtest_machines, MachineSoundDataset
from dcase2020.config import DATA_ROOT
from dcase2020.datasets.preprocessing import baseline_preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

preprocessing_params = {
    "sr": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "power": 2.0,
    "n_mels": 128
}

dev_test_loader_list = []
dev_test_classes = []
dev_test_total = []
dev_test_pos = []
for subset in all_devtest_machines.keys():
    # subset_path = "dev/{}/test".format(_config["machine_type"]
    for mid in all_devtest_machines[subset]:
        dev_test = MachineSoundDataset(root_dir=DATA_ROOT, subset_path=subset,
                                       machine_id=mid,
                                       preprocessing_fn=baseline_preprocessing,
                                       preprocessing_name="baseline_preprocessing",
                                       preprocessing_params=preprocessing_params,
                                       use_input_as_target=False,
                                       use_machine_id_as_target=False,
                                       use_machine_type_as_target=False,
                                       maximum_snippets="auto",
                                       frames_per_snippet=8,
                                       return_meta=True,
                                       norm_std_mean=False)

        dev_test_classes.append(subset+"_"+mid)
        dev_test_pos.append(sum(dev_test.labels))
        dev_test_total.append(len(dev_test.labels))
        # dev_test.nr_classes = 6
        # dev_test_loader = DataLoader(dev_test, batch_size=int(dev_test.maximum_snippets), shuffle=False)
        # dev_test_loader_list.append(dev_test_loader)


dev_test_pos_perc = [pos / tot for (pos, tot) in zip(dev_test_pos, dev_test_total)]


x = range(len(dev_test_classes))
# total
plt.bar(x=x, height=dev_test_total)
plt.bar(x=x, height=dev_test_pos)
plt.xticks(ticks=x, labels=dev_test_classes, rotation=90)
plt.legend(["normal", "anomalous"])
plt.show()

# percentage
plt.bar(x=range(len(dev_test_classes)), height=1)
plt.bar(x=range(len(dev_test_classes)), height=dev_test_pos_perc)
plt.xticks(ticks=x, labels=dev_test_classes, rotation=90)
plt.legend(["normal", "anomalous"])
plt.show()