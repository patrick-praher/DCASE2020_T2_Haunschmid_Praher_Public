from results.summarize_results_in_db import collect_postprocessed_results, extract_exp_data_flows
from dcase2020.secret_config import mongo_password
from dcase2020.config import mongo_connection_string
from dcase2020.config import experiments_path
import pandas as pd
import matplotlib.pyplot as plt

## data unintentionally stored in wrong db!!

db_name = 'dcase2020_task2_flows_maf'
experiment_data_maf = extract_exp_data_flows(mongo_db=mongo_connection_string.format(mongo_password),
                                         db_name=db_name)

run_ids = list(range(197, max(experiment_data_maf['run_id']) + 1))
experiment_data_maf = experiment_data_maf[experiment_data_maf["run_id"].isin(run_ids)]
results_maf = collect_postprocessed_results(experiments_path, db_name, run_ids, False)

experiment_data_maf["run_id"] = [i+1000 for i in experiment_data_maf["run_id"]]
results_maf["run_id"] = [i+1000 for i in results_maf["run_id"]]

# update run_ids in both thingis

###

db_name = 'dcase2020_task2_flows_grid'
OVERWRITE = False

## build plot_data_flow_run_id using EvaluationResults & correct postprocessing
# plot_data_flow_run_id = extract_plot_data_flows(mongo_db_uri, aggregate=False) ## TODO remove this (duplicate info)
experiment_data = extract_exp_data_flows(mongo_db=mongo_connection_string.format(mongo_password),
                                         db_name=db_name)
run_ids = experiment_data['run_id'].unique()

results = collect_postprocessed_results(experiments_path, db_name, run_ids, OVERWRITE)

# combinde date with wrong data in flows_maf
experiment_data = pd.concat([experiment_data, experiment_data_maf], ignore_index=True)
results = pd.concat([results, results_maf], ignore_index=True)

successful_run_ids = results['run_id'].unique()

results_by_mtype = results[['run_id', 'machine_type', 'rocauc', 'p_rocauc']].groupby(['run_id', 'machine_type']).aggregate(['mean']).reset_index()
# results_by_mtype = plot_data_flow_run_id[['run_id', 'machine_type', 'rocauc', 'p_rocauc']].groupby(['run_id', 'machine_type']).aggregate(['mean']).reset_index()
NUM_EXPERIMENTS = len(successful_run_ids)

rocauc_by_mtype = results_by_mtype[['run_id', 'machine_type', 'rocauc']]
rocauc_by_mtype.columns=['run_id', 'machine_type', 'rocauc']
rocauc_by_mtype = rocauc_by_mtype.sort_values(by=['machine_type', 'rocauc'], ascending=False)
rocauc_by_mtype['r_rank'] = list(range(1,NUM_EXPERIMENTS+1)) * 6

p_rocauc_by_mtype = results_by_mtype[['run_id', 'machine_type', 'p_rocauc']]
p_rocauc_by_mtype.columns=['run_id', 'machine_type', 'p_rocauc']
p_rocauc_by_mtype = p_rocauc_by_mtype.sort_values(by=['machine_type', 'p_rocauc'], ascending=False)
p_rocauc_by_mtype['p_rank'] = list(range(1,NUM_EXPERIMENTS+1)) * 6

metrics_merged = pd.merge(p_rocauc_by_mtype, rocauc_by_mtype)
metrics_merged['avg_rank'] = (metrics_merged['p_rank'] + metrics_merged['r_rank']) / 2

# TODO: same rank, higher pAUC wins!

experiments_ranking = metrics_merged[['run_id', 'avg_rank']].groupby('run_id', as_index = False).aggregate('mean').sort_values('avg_rank', ascending=True)

ranking_arch_params = pd.merge(experiment_data, experiments_ranking, left_on='run_id', right_on='run_id').sort_values('avg_rank', ascending=True)

plt.figure()
plt.scatter(ranking_arch_params['valid_loss'], ranking_arch_params['avg_rank'], c=ranking_arch_params['frames_per_snippet'])
plt.show()

colors = ['blue' if t=='realnvp' else 'orange' for t in ranking_arch_params['arch_params.flow_model_type']]
sizes = [s*5 for s in ranking_arch_params['arch_params.n_hidden']]
plt.figure()
plt.scatter(ranking_arch_params['arch_params.hidden_size'], ranking_arch_params['avg_rank'], c=colors, s=sizes)
plt.show()


## compare models based on norm_per_set

subset = ranking_arch_params.loc[
    (ranking_arch_params['arch_params.flow_model_type']=='maf') &
    (ranking_arch_params['arch_params.cond_label_size']==6) &
    (ranking_arch_params['arch_params.hidden_size']==512) &
    (ranking_arch_params['optimizer_params.lr']==0.0001) &
    (ranking_arch_params['frames_per_snippet']==2) &
    (ranking_arch_params['batch_size']==2048) &
    (ranking_arch_params['arch_params.n_blocks'] < 6)
    ]

