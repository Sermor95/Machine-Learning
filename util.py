import logging
from functions import *


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def logging_info(function,info):
    logging.info(f'//---{function}--->{info}---//')

def get_init_config():
    configs = []

    datasets = get_datasets()
    cribas = [0.9,0.8,0.7]
    reductions = [10,30,50]
    models = ['decision-tree','random-forest','gradient-boosting']

    for ds in datasets:
        for c in cribas:
            for r in reductions:
                for m in models:
                    configs.append({'dataset': ds,
                                    'criba': c,
                                    'reduction': r,
                                    'model': m,
                                    'launchers': 10})
    return configs

