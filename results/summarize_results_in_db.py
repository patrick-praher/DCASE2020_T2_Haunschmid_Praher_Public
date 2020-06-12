from incense import ExperimentLoader
import pandas as pd
from dcase2020.secret_config import mongo_password
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dcase2020.utils import  pickle_load, pickle_dump
from results.plot_snippet_loss import extract_model_outputs
from dcase2020.config import mongo_connection_string
mongo_db_uri = mongo_connection_string.format(mongo_password)

import numpy as np

from dcase2020.datasets.machine_sound_dataset import all_devtest_machines

def aggregate_data(results, group_by_args, aggregate):

    plot_data = None

    if(aggregate):
        plot_data = results.groupby(group_by_args).aggregate(['mean', 'std', 'count']).reset_index()
        plot_data.columns = [' '.join(col).strip() for col in plot_data.columns.values]
    else:
        plot_data = results

    return plot_data


def extract_plot_data_baseline(mongo_db, aggregate = True):
    db_name = 'dcase2020_task2_baseline'
    loader = ExperimentLoader(
        mongo_uri=mongo_db,
        db_name=db_name
    )

    experiments = loader.find(query={"status":"COMPLETED", "_id": {"$lt": 329}})  # at 329 metrics were renamed

    # arch_params might be different for different experiments
    # results per channel multiplier should be compared to baseline
    results = pd.DataFrame([(e1.config['machine_type'], e1.config['machine_id'], e1.metrics['eval_rocauc'][0] * 100, e1.metrics['eval_p_rocauc'][0] * 100) for e1 in experiments])
    results = results.rename(columns={0:'machine_type', 1:'machine_id', 2:'rocauc', 3:'p_rocauc'})

    group_by_args = ['machine_type', 'machine_id']
    selected_columns = copy.deepcopy(group_by_args)
    selected_columns.extend(['rocauc', 'p_rocauc'])

    results = results[results.columns.intersection(selected_columns)]

    plot_data = aggregate_data(results, group_by_args, aggregate)

    plot_data['ID'] = plot_data[['machine_type', 'machine_id']].apply(lambda x: ' '.join(x), axis=1)
    plot_data['Class'] = 'Baseline'

    return plot_data


def extract_plot_data_conv_ae(mongo_db, aggregate = True):
    db_name = 'dcase_task2_fully_conv_ae'
    loader = ExperimentLoader(
        mongo_uri=mongo_db,
        db_name=db_name
    )

    experiments = loader.find(query={"status":"COMPLETED", "config.num_epochs":100})

    # e1.config['preprocessing_params']['n_mels'], <- not changed in this experiment
    # arch_params might be different for different experiments
    # results per channel multiplier should be compared to baseline
    results = pd.DataFrame([(e1.config['machine_type'], e1.config['machine_id'], e1.config['arch_params']['channel_multiplier'], e1.config.get('apply_normalization', False), e1.metrics['eval_rocauc'][0] * 100, e1.metrics['eval_p_rocauc'][0] * 100) for e1 in experiments])
    results = results.rename(columns={0:'machine_type', 1:'machine_id', 2:'channel_multiplier', 3:'apply_normalization', 4:'rocauc', 5:'p_rocauc'})

    group_by_args = ['machine_type', 'machine_id', 'channel_multiplier', 'apply_normalization']
    selected_columns = copy.deepcopy(group_by_args)
    selected_columns.extend(['rocauc', 'p_rocauc'])

    results = results[results.columns.intersection(selected_columns)]

    plot_data = aggregate_data(results, group_by_args, aggregate)

    plot_data['ID'] = plot_data[['machine_type', 'machine_id']].apply(lambda x: ' '.join(x), axis=1)

    plot_data['channel_multiplier'] = plot_data['channel_multiplier'].astype(str)
    plot_data['apply_normalization'] = plot_data['apply_normalization'].astype(str)

    plot_data['Class'] = plot_data[['channel_multiplier', 'apply_normalization']].apply(lambda x: ' '.join(x), axis=1)
    plot_data['Class'] = plot_data['Class'] + "_" + db_name

    return plot_data


def extract_exp_data_flows(mongo_db, db_name, run_id=None):
    loader = ExperimentLoader(
        mongo_uri=mongo_db,
        db_name=db_name
    )

    if(run_id is not None):
        experiments = loader.find({"status": "COMPLETED", "_id":run_id})
    else:
        experiments = loader.find({"status": "COMPLETED"})

    col_names = ['run_id',
                 'apply_normalization',
                 'norm_per_set',
                 'transpose_flatten',
                 'frames_per_snippet',
                 'batch_size',
                 'arch_params.n_hidden',
                 'fixed_flow_evaluation',
                 'arch_params.hidden_size',
                 'arch_params.n_blocks',
                 'arch_params.flow_model_type',
                 'arch_params.cond_label_size',
                 'optimizer',
                 'optimizer_params.lr',
                 'status',
                 'valid_loss']

    results = pd.DataFrame(columns=col_names)

    for e in experiments:
        print("Processing experiment {}".format(e.id))
        losses = e.metrics['valid_loss']
        new_row = pd.DataFrame([[
                     e.id,
                     e.config['apply_normalization'],
                     e.config.get('norm_per_set', False),
                     e.config.get('transpose_flatten', False),
                     e.config['frames_per_snippet'],
                     e.config['batch_size'],
                     e.config['arch_params']['n_hidden'],
                     e.config.get('fixed_flow_evaluation', False),
                     e.config['arch_params']['hidden_size'],
                     e.config['arch_params']['n_blocks'],
                     e.config['arch_params']['flow_model_type'],
                     e.config['arch_params'].get('cond_label_size', 6),
                     e.config['optimizer'],
                     e.config['optimizer_params']['lr'],
                     e.status,
                     losses[len(losses)-1-e.config.get('early_stopping_patience', 0)]]],
                   columns=col_names)
        results = results.append(new_row)

    return results


def extract_plot_data_flows(mongo_db, aggregate = True, grouping_vars=None):
    db_name = 'dcase2020_task2_flows_maf'
    loader = ExperimentLoader(
        mongo_uri=mongo_db,
        db_name=db_name
    )

    # experiments = loader.find_all()
    experiments = loader.find({"status": "COMPLETED"})

    col_names = ['run_id', 'machine_type', 'machine_id', 'rocauc', 'p_rocauc', 'apply_normalization', 'frames_per_snippet',
                 'n_hidden', 'fixed_flow_evaluation','hidden_size', 'n_blocks', 'optimizer', 'optimizer_params.lr',
                 'status']

    if grouping_vars is None:
        grouping_vars = col_names[5:]

    results = pd.DataFrame(columns=col_names)

    for e in experiments:
        for type in all_devtest_machines.keys():
            for id in all_devtest_machines[type]:
                print(e.id, type + '_' + id + "_rocauc")

                rocauc = e.metrics.get(type + '_' + id + "_rocauc",[0])[0]
                if rocauc is not None:
                    rocauc = rocauc * 100
                p_rocauc = e.metrics.get(type + '_' + id + "_p_rocauc", [0])[0]
                if p_rocauc is not None:
                    p_rocauc = p_rocauc * 100

                if rocauc is None or p_rocauc is None:
                    print("Skipping experiment", e.id)
                    continue

                new_row = pd.DataFrame([[e.id,
                                         type.split('/')[1],
                                         id,
                                         rocauc,
                                         p_rocauc,
                                         e.config['apply_normalization'],
                                         e.config['frames_per_snippet'],
                                         e.config['arch_params']['n_hidden'],
                                         e.config.get('fixed_flow_evaluation', False),
                                         e.config['arch_params']['hidden_size'],
                                         e.config['arch_params']['n_blocks'],
                                         e.config['optimizer'],
                                         e.config['optimizer_params']['lr'],
                                         e.status]],
                                       columns=col_names)
                results = results.append(new_row)

    group_by_args = ['run_id', 'machine_type', 'machine_id']
    group_by_args.extend(grouping_vars)

    selected_columns = copy.deepcopy(group_by_args)
    selected_columns.extend(['rocauc', 'p_rocauc'])

    results = results[results.columns.intersection(selected_columns)]

    plot_data = aggregate_data(results, group_by_args, aggregate)

    plot_data['ID'] = plot_data[['machine_type', 'machine_id']].apply(lambda x: ' '.join(x), axis=1)

    for var in grouping_vars:
        plot_data[var] = plot_data[var].astype(str)

    plot_data['Class'] = plot_data[grouping_vars].apply(lambda x: ' '.join(x), axis=1)
    plot_data['Class'] = plot_data['Class'] + "_" + db_name

    return plot_data


def plot_data_baseline_postprocessed():
    data = pd.DataFrame(
        np.array(
            [['ToyCar', 'id_01', 82.16],
             ['ToyCar', 'id_02', 85.31],
             ['ToyCar', 'id_03', 64.36],
             ['ToyCar', 'id_04', 86.10],
             ['ToyConveyor', 'id_01', 79.23],
             ['ToyConveyor', 'id_02', 64.88],
             ['ToyConveyor', 'id_03', 76.61],
             ['fan', 'id_00', 54.38],
             ['fan', 'id_02', 74.88],
             ['fan', 'id_04', 61.9],
             ['fan', 'id_06', 74.04],
             ['pump', 'id_00', 66.76],
             ['pump', 'id_02', 64.16],
             ['pump', 'id_04', 88.47],
             ['pump', 'id_06', 75.21],
             ['slider', 'id_00', 98.12],
             ['slider', 'id_02', 76.93],
             ['slider', 'id_04', 97.83],
             ['slider', 'id_06', 92.90],
             ['valve', 'id_00', 88.76],
             ['valve', 'id_02', 91.64],
             ['valve', 'id_04', 92.1],
             ['valve', 'id_06', 68.49]
             ]),
        columns=['machine_type', 'machine_id', 'rocauc'])
    data["ID"] = data[['machine_type', 'machine_id']].apply(lambda x: ' '.join(x), axis=1)
    data["Class"] = "Baseline + Postprocessing"

    return data

def plot_data_baseline_from_website():
     data = pd.DataFrame(
        np.array(
            [['ToyCar', 'id_01', 81.36],
             ['ToyCar', 'id_02', 85.97],
             ['ToyCar', 'id_03', 62.3],
             ['ToyCar', 'id_04', 84.45],
             ['ToyConveyor', 'id_01', 78.07],
             ['ToyConveyor', 'id_02', 64.16],
             ['ToyConveyor', 'id_03', 75.35],
             ['fan', 'id_00', 54.41],
             ['fan', 'id_02', 73.4],
             ['fan', 'id_04', 61.61],
             ['fan', 'id_06', 73.92],
             ['pump', 'id_00', 67.15],
             ['pump', 'id_02', 61.53],
             ['pump', 'id_04', 88.33],
             ['pump', 'id_06', 74.55],
             ['slider', 'id_00', 96.19],
             ['slider', 'id_02', 78.97],
             ['slider', 'id_04', 94.30],
             ['slider', 'id_06', 69.59],
             ['valve', 'id_00', 68.76],
             ['valve', 'id_02', 68.18],
             ['valve', 'id_04', 74.30],
             ['valve', 'id_06', 53.90]
             ]),
        columns=['machine_type', 'machine_id', 'rocauc'])
     data["ID"] = data[['machine_type', 'machine_id']].apply(lambda x: ' '.join(x), axis=1)
     data["Class"] = "Baseline Website"

     return data

def plot_experiments_bars(plot_data, measure = "rocauc mean"):
    # seaborn plot
    sns.set(style="whitegrid")
    g = sns.catplot(x="ID", y=measure, hue="Class", data=plot_data, height=6, kind="bar", palette="muted", aspect=2)
    g.set_xticklabels(rotation=90)
    g.set(ylim=(50, 100))
    plt.show()


def collect_postprocessed_results(experiments_path, db_name, run_ids, overwrite=False):
    columns = ['machine', 'machine_type_id', 'machine_type', 'machine_id', 'metric', 'rocauc', 'p_rocauc']
    results = pd.DataFrame(columns=columns)
    successful_run_ids = []

    for id in run_ids:
        prestored_path = os.path.join(experiments_path, 'postprocessed', db_name, str(id).zfill(4) + ".pt")
        tmp_results = pd.DataFrame(columns=columns)
        if not os.path.exists(prestored_path) or overwrite:
            data = extract_model_outputs(mongo_db_uri, db_name, run_id=id)
            for k in data.keys():
                tmp_results = tmp_results.append(data[k].get_metric(k))
            pickle_dump(tmp_results, prestored_path)
        else:
            tmp_results = pickle_load(prestored_path)
        if len(tmp_results) == 23:
            successful_run_ids.append(id)
            results = results.append(tmp_results)
        elif len(tmp_results) > 0:
            print("run_id {} has {} results", id, len(tmp_results))
    print(len(successful_run_ids) * 23)
    print(len(results))
    results['run_id'] = np.repeat(successful_run_ids, 23)
    return results

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--db_name", type=str)
    # # parser.add_argument("--group_by", type=str, nargs="+")
    # args = parser.parse_args()
    # db_name = args.db_name
    #
    # #db_name = 'dcase_task2_fully_conv_ae'

    # mongo_db_uri = 'mongodb+srv://mongoAtlasUser:{}@cluster0-brzcp.mongodb.net/?retryWrites=true&w=majority'.format(mongo_password)

    plot_data_conv_ae = extract_plot_data_conv_ae(mongo_db_uri)
    plot_data_b = extract_plot_data_baseline(mongo_db_uri)
    plot_data_flow = extract_plot_data_flows(mongo_db_uri)

    # baseline vs. conv_ae
    plot_data = plot_data_conv_ae[['Class', 'ID', 'rocauc mean']]
    plot_data = plot_data.append(plot_data_b[['Class', 'ID', 'rocauc mean']])

    plot_experiments_bars(plot_data)


    plot_data_b_online = plot_data_baseline_from_website()


    # baseline vs. flow
    plot_data_b = extract_plot_data_baseline(mongo_db_uri, aggregate=False)
    plot_data_flow = extract_plot_data_flows(mongo_db_uri, aggregate=False, grouping_vars = ['n_hidden'])

    plot_data = plot_data_b[['Class', 'ID', 'rocauc']]
    plot_data = plot_data.append(plot_data_b_online)
    plot_data = plot_data.append(plot_data_flow[['Class', 'ID', 'rocauc']])


    plot_experiments_bars(plot_data, measure="rocauc")

    tmp = extract_exp_data_flows(mongo_db=mongo_db_uri, db_name='dcase2020_task2_flows_maf')

    # g = sns.catplot(x="ID", hue="channel_multiplier", data=summary2, height=6, kind="count", palette="muted")
    # plt.show(g)

# TESTING
# from incense import ExperimentLoader
# mongo_db_uri = 'mongodb+srv://mongoAtlasUser:{}@cluster0-brzcp.mongodb.net/?retryWrites=true&w=majority'.format(mongo_password)
# db_name = 'dcase2020_task2_flows_maf'
# db_name = 'dcase2020_task2_baseline'
#
# loader = ExperimentLoader(
#     mongo_uri=mongo_db_uri,
#     db_name=db_name
# )


