from argparse import ArgumentParser
from dcase2020.utils import pickle_load
from dcase2020.flow_predictor import FlowPredictor

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=int, required=True)
    args = parser.parse_args()

    preds = pickle_load("eval_preds/{}.pt".format(args.model_id))

    FlowPredictor.save_csv(args.model_id, preds)






