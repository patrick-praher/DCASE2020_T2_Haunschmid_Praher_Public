# change configuration using hostname/username if required
import socket
import getpass
import os

# add normalizing_flows to paths
deployment_path = "~/deployment/"
deployment_path = os.path.expanduser(deployment_path)
experiments_path = "~/experiments/dcase2020/"
experiments_path = os.path.expanduser(experiments_path)

mongo_connection_string = ""

CACHE_DIR = '~/data/cached_datasets/dcase2020/task2/'
DATA_ROOT = '/share/cp/datasets/DCASE2020/task2/'


