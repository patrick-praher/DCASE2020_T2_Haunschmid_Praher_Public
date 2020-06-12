from incense import ExperimentLoader
from torch.utils.data import DataLoader
from dcase2020.datasets.machine_sound_dataset import prepare_batch
from dcase2020.trainer import adapt_dimension
from dcase2020.model_architectures.flows import FlowMAFModel
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import time

from dcase2020.datasets.machine_sound_dataset import MachineSoundDataset, all_evaltest_machines, \
    machine_type_to_class_label, onehot
from dcase2020.config import DATA_ROOT
from dcase2020.datasets.preprocessing import baseline_preprocessing
from dcase2020.utils import pickle_load, pickle_dump

from dcase2020.secret_config import mongo_password
from dcase2020.config import mongo_connection_string

from os import path
import csv
import warnings
import os

from dcase2020.postprocess import postprocess, type_metric_mapping


class FlowPredictor:

    def __init__(self, mongo_db_uri, db_name, model_path, run_id):

        # load experiment
        loader = ExperimentLoader(
            mongo_uri=mongo_db_uri,
            db_name=db_name
        )

        self.preds = None
        self.run_id = run_id
        self.e = loader.find_by_id(run_id)

        self.model_path = path.join(model_path, self.e.config['uid'], 'best_model.ckpt')

    def __load_data(self):

        set_stats = pickle_load('stats/allstats_nps{}.pt'.format(int(self.e.config['norm_per_set'])))
        # load data
        self.eval_test_loader_list = []
        for subset in all_evaltest_machines.keys():
            for mid in all_evaltest_machines[subset]:
                set_stats_name = subset.replace("test", "train") + "_" + mid
                print("Loading", set_stats_name)
                eval_test = MachineSoundDataset(root_dir=DATA_ROOT, subset_path=subset,
                                                machine_id=mid,
                                                preprocessing_fn=baseline_preprocessing,
                                                preprocessing_name="baseline_preprocessing",
                                                preprocessing_params=self.e.config['preprocessing_params'],
                                                use_input_as_target=False,
                                                use_machine_id_as_target=False,
                                                use_machine_type_as_target=False,
                                                maximum_snippets=self.e.config['maximum_snippets'],
                                                frames_per_snippet=self.e.config['frames_per_snippet'],
                                                return_meta=True,
                                                norm_std_mean=self.e.config["apply_normalization"],
                                                mean=set_stats[set_stats_name][0],
                                                std=set_stats[set_stats_name][1]
                                                )

                eval_test.nr_classes = self.e.config['arch_params']['cond_label_size']
                eval_test_loader = DataLoader(eval_test, batch_size=int(eval_test.maximum_snippets), shuffle=False,
                                              pin_memory=True, num_workers=10)
                self.eval_test_loader_list.append(eval_test_loader)

    def __load_model(self):

        self.model = FlowMAFModel(flow_model_type=self.e.config['arch_params']['flow_model_type'],
                                  n_blocks=self.e.config['arch_params']['n_blocks'],
                                  input_size=self.e.config['frames_per_snippet'] *
                                             self.e.config['preprocessing_params']['n_mels'],
                                  hidden_size=self.e.config['arch_params']['hidden_size'],
                                  n_hidden=self.e.config['arch_params']['n_hidden'],
                                  cond_label_size=self.e.config['arch_params'].get('cond_label_size', 6),
                                  input_order=self.e.config['arch_params']['input_order'],
                                  use_batch_norm=self.e.config['arch_params']['flow_use_batch_norm'])

        S = torch.load(self.model_path)
        self.model.load_state_dict(S['model_state_dict'])
        self.model.eval()
        self.model.cuda()

    @torch.no_grad()
    def calc_predictions(self):

        self.__load_model()
        self.__load_data()

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        preds = {}

        start_time = time.time()
        for dl in self.eval_test_loader_list:
            ds = dl.dataset
            print("Evaluating {}/{}".format(ds.subset_path, ds.machine_id))
            machine = ds.subset_path + "_" + ds.machine_id
            preds[machine] = []

            for batch in dl:
                x, meta, y = prepare_batch(batch, device)

                mt = ds.subset_path.split("/")[1]
                class_id = machine_type_to_class_label[mt]
                y_flow = onehot(class_id, ds.nr_classes)
                y_flow = adapt_dimension(y_flow, "unsqueeze").to(device)

                # compute anomaly_score per snippet
                if self.e.config.get("transpose_flatten", False):
                    x = adapt_dimension(x, "transpose_flatten")
                else:
                    x = adapt_dimension(x, "flatten")

                anomaly_score = self.model.anomaly_score(x, y_flow, nn.MSELoss(reduction="none"))  # .mean(0)
                sample_names = meta[2]

                metric = type_metric_mapping[mt]
                if (metric == "50%"):
                    prediction = anomaly_score.median().item()
                elif (metric == "std"):
                    prediction = anomaly_score.std().item()
                else:
                    warnings.warn("No fitting metric found: {}".format(metric))

                preds[machine].append((sample_names[0], prediction))

                assert all(x == sample_names[0] for x in sample_names), "Not all samples names in batch are equal!"
            print("Time passed:", time.time() - start_time)

        self.preds = preds

    @staticmethod
    def save_csv(run_id, preds):
        folder = "eval_preds/{}/".format(run_id)
        try:
            os.mkdir(folder)
        except OSError:
            print("Creation of the directory %s failed" % folder)
        else:
            for k in preds.keys():

                filename = os.path.join(folder,
                                        "anomaly_score_" + k.split("/")[1] + "_" + k.split("/")[2].replace("test_",
                                                                                                           "") + ".csv")
                print(filename)

                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for line in preds[k]:
                        writer.writerow(line)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=int, required=True)
    parser.add_argument("--model_dir", type=str, default="/workspace/data/trained_model/")
    parser.add_argument("--results_db", type=str, default="dcase2020_task2_flows_grid")
    args = parser.parse_args()

    pred = FlowPredictor(
        mongo_connection_string.format(mongo_password),
        args.results_db,
        args.model_dir,
        args.model_id
    )

    pred.calc_predictions()
    FlowPredictor.save_csv(pred.run_id, pred.preds)

    # pickle_dump(preds, "eval_preds/{}.pt".format(args.model_id))
    # model_path = "/run/user/1000/gvfs/smb-share:server=10.0.0.55,share=daten/DCASE/trained_model/2020-6-2_8-54-33.449015/best_model.ckpt"
