import os
import pickle
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from incense import ExperimentLoader

from dcase2020.evaluation_results import EvaluationResults
from dcase2020.secret_config import mongo_password
from dcase2020.config import mongo_connection_string

def extract_model_outputs(mongo_db, db_name, run_id=None):
    loader = ExperimentLoader(
        mongo_uri=mongo_db,
        db_name=db_name
    )
    if run_id is None:
        experiments = loader.find(query={"status": "COMPLETED"})
    else:
        experiments = [loader.find_by_id(run_id)]

    model_output = {}
    for e in experiments:
        for k in e.artifacts.keys():

            if 'evaluation_results' in k:
                art = e.artifacts[k]
                art.save(tempfile.gettempdir())
                tmp_filename = os.path.join(tempfile.gettempdir(),art._make_filename())
                exp_res: EvaluationResults = pickle.load(open(tmp_filename, 'rb'))
                os.remove(tmp_filename)

                model_output[exp_res.subset_path + "_" +
                             exp_res.machine_id + "_" +
                             str(exp_res.start_time) + "_" +
                             str(e.id)] = exp_res

    return model_output


if __name__ == '__main__':
    mongo_db_uri = mongo_connection_string.format(mongo_password)

    data = extract_model_outputs(mongo_db_uri, "dcase2020_task2_flows_maf")

    sns.set(style="whitegrid")
    df = data['dev/slider/test_id_00_2020-05-11 14:21:50.506153'].snippets
    df = pd.melt(df, id_vars=['ID'])
    g = sns.FacetGrid(df, col="ID", col_wrap=5, height=1.5)
    g = g.map(plt.plot, "variable", "value", marker=".")
    plt.show()
