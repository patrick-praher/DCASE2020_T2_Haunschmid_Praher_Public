import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

type_metric_mapping = {"ToyCar": "50%", "ToyConveyor": "50%", "fan": "50%",
                       "pump": "50%", "slider": "std", "valve": "std"}

class EvaluationResults:
    subset_path = None
    machine_id = None
    start_time = None
    snippets = None

    def __init__(self, s_p, m_id, s_t, nr_snippets):
        self.subset_path = s_p
        self.machine_id = m_id
        self.start_time = s_t
        self.snippets = pd.DataFrame(columns=['ID'] + list(map(str, range(nr_snippets))))

    def add_sample(self, name, snips):
        new_row = pd.DataFrame([[name] + snips],
                               columns=self.snippets.columns)
        self.snippets = self.snippets.append(new_row)

    def get_metric(self, id, metric=None):
        if metric is None:
            print("Selecting default metric: ")
            metric = type_metric_mapping[id.split('/')[1]]
            print(metric)

        metrics = self.get_all_metrics(id)
        return metrics.loc[metrics['metric'] == metric]

    def get_all_metrics(self, id):

        df = self.snippets.melt(id_vars=['ID'])
        agg = df.groupby(["ID"]).describe().reset_index()
        agg.columns = agg.columns = [' '.join(col).strip() for col in agg.columns.values]

        agg["label"] = [1 if 'anomaly' in el else 0 for el in agg["ID"]]

        columns = ['machine', 'machine_type_id', 'machine_type', 'machine_id', 'metric', 'rocauc', 'p_rocauc']
        results = pd.DataFrame(columns=columns)

        for c in np.setdiff1d(agg.columns.tolist(), ['ID', 'value count', 'label']):
            nice_key = c.split(' ')[1]
            rocauc = roc_auc_score(agg['label'].tolist(), agg[c].tolist())
            p_rocauc = roc_auc_score(agg['label'].tolist(), agg[c].tolist(), max_fpr=0.1)

            machine = self.subset_path + "_" + self.machine_id
            row = pd.DataFrame([[machine + "_" + str(self.start_time) + "_" + str(id),
                                 machine,
                                 self.subset_path,
                                 self.machine_id,
                                 nice_key,
                                 rocauc,
                                 p_rocauc]],
                               columns=columns)
            results = results.append(row)

        return (results)




