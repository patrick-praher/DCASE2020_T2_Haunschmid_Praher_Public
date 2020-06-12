from incense import ExperimentLoader
import pandas as pd

loader = ExperimentLoader(
    mongo_uri='rechenknecht2.cp.jku.at:37373',
    db_name='dcase2020_task2_baseline_v'
)

experiments = loader.find(query={"status":"COMPLETED"})
results = pd.DataFrame([(e1.config['machine_type'], e1.config['machine_id'], e1.config['preprocessing_params']['n_mels'], e1.metrics['eval_rocauc'][0] * 100, e1.metrics['eval_p_rocauc'][0] * 100) for e1 in experiments])
results = results.rename(columns={0:'machine_type', 1:'machine_id', 2:'n_mels', 3:'rocauc', 4:'p_rocauc'})

summary = results.groupby(['machine_type', 'machine_id', 'n_mels']).aggregate(['mean', 'std', 'count'])
