import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from dcase2020.datasets.machine_sound_dataset import prepare_batch, machine_type_to_class_label, onehot
from dcase2020.evaluation_results import EvaluationResults
from dcase2020.config import experiments_path
import tempfile
import datetime
import pickle
import os
import math


def initialize_optimizer(model, optimizer_name, optimizer_params):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    else:
        raise ValueError("{} is not a valid optimizer".format(optimizer_name))
    return optimizer


def adapt_dimension(x, mode):
    assert mode in ['flatten', 'unsqueeze', 'transpose_flatten']
    if mode == 'flatten':
        x = x.view(x.size(0), -1)
    elif mode == 'unsqueeze':
        x = x.unsqueeze(1)
    elif mode == 'transpose_flatten':
        x = x.t().flatten()
    return x


class BaseTrainer:
    def __init__(self, experiment, model, train_loader, validation_loader, config, target_mode, transpose_flatten=False):
        self.experiment = experiment
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.target_mode = target_mode
        self.transpose_flatten = False
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)
        self.optimizer = initialize_optimizer(model, config['optimizer'], config['optimizer_params'])
        self.criterion = nn.MSELoss()  # optimization criterion (averaged over batch)
        self.loss = nn.MSELoss(reduction="none")  # loss used for computing anomaly score
        self.train_loss = []
        self.validation_loss = []
        self.cond_label_size = config['arch_params'].get('cond_label_size', None)

        self.early_stopping_patience = self.config.get("early_stopping_patience", None)
        self.early_stopping_min_delta = self.config.get("early_stopping_min_delta", 0.0)
        self.model_dir = os.path.join(experiments_path, self.config['uid'])
        self.best_model_path = os.path.join(self.model_dir, 'best_model.ckpt')
        self.best_model_loss = np.Inf

    def is_better_than_current_best(self, valid_loss):
        if len(self.validation_loss) == 0:
            return True
        return valid_loss < (self.best_model_loss - self.early_stopping_min_delta)

    def load_best_model(self):
        print("Loading best model ...")
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print("Loaded model from epoch {} with loss {}".format(epoch, loss))
            return True
        return False

    def train(self, num_epochs):
        """Train model using training set."""

        best_epoch = 0
        check_epochs = num_epochs
        didnt_improve_since = 0
        if self.early_stopping_patience is not None:
            check_epochs = self.early_stopping_patience

        try:
            for epoch in range(num_epochs):
                print("Starting epoch {}".format(epoch))

                self.model.train()
                train_loss = self.train_(epoch)
                if np.isnan(train_loss):
                    train_loss = np.Inf
                self.experiment.log_scalar("train_loss", train_loss)
                self.train_loss.append(train_loss)

                self.model.eval()
                valid_loss = self.validate_(epoch)
                if np.isnan(valid_loss):
                    valid_loss = np.Inf

                print("epoch {}: train_loss: {} valid_loss: {}".format(epoch, train_loss, valid_loss))
                # check if has improved
                is_better = self.is_better_than_current_best(valid_loss)

                self.validation_loss.append(valid_loss)
                self.experiment.log_scalar("valid_loss", valid_loss)

                if is_better and epoch > 0 and (not np.isinf(valid_loss) and not np.isinf(train_loss)):
                    self.best_model_loss = valid_loss
                    # model from epoch 0 does not work
                    didnt_improve_since = 0
                    if not os.path.exists(self.model_dir):
                        os.mkdir(self.model_dir)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': valid_loss
                    }, self.best_model_path)
                else:
                    didnt_improve_since += 1

                print("Finished epoch", epoch)

                if self.early_stopping_patience is not None and didnt_improve_since >= self.early_stopping_patience:
                    print("Early stopping ...")
                    break

                if np.isinf(train_loss) or np.isinf(valid_loss):
                    print("Stopping because loss is inf ... train_loss: {} valid_loss: {}".format(train_loss, valid_loss))
                    break
        except KeyboardInterrupt:
            pass
            # print("Caught KeyboardInterrupt, trying to evaluate ...")
            # trainer.load_best_model()

        loaded_success = self.load_best_model()
        if not loaded_success:
            raise Exception("No working model stored.")

    def train_(self, epoch):
        raise NotImplementedError("You need to implement train_!")

    def validate_(self, epoch):
        raise NotImplementedError("You need to implement validate_!")

    def evaluate(self, evaluation_loaders, save_eval_results=False):
        if not isinstance(evaluation_loaders, list):
            evaluation_loaders = [evaluation_loaders]

        self.model.eval()

        print("length eval loader:", len(evaluation_loaders))

        for evaluation_loader in evaluation_loaders:
            labels = np.zeros(len(evaluation_loader))
            predictions = np.zeros(len(evaluation_loader))

            ds = evaluation_loader.dataset

            print(ds.subset_path)
            i = 0
            eval_result = EvaluationResults(ds.subset_path,
                                            ds.machine_id,
                                            datetime.datetime.now(),  # start_time of experiment can't be accessed
                                            ds.maximum_snippets)

            for batch in evaluation_loader:
                x, meta, y = prepare_batch(batch, self.device)
                y_target = y.sum() / len(y)
                assert y_target == 0 or y_target == 1, 'something is wrong with your labels'
                class_id = machine_type_to_class_label[ds.subset_path.split("/")[1]]
                y_flow = onehot(class_id, ds.nr_classes)
                y_flow = adapt_dimension(y_flow, self.target_mode).to(self.device)

                # compute anomaly_score per snippet
                if self.transpose_flatten:
                    x = adapt_dimension(x, "transpose_flatten")
                else:
                    x = adapt_dimension(x, "flatten")
                anomaly_score = self.model.anomaly_score(x, y_flow, self.loss)#.mean(0)
                sample_names = meta[2]
                assert all(x == sample_names[0] for x in sample_names), "Not all samples names in batch are equal!"

                # add anomaly_score per snippet to evaluation result
                eval_result.add_sample(sample_names[0], anomaly_score.tolist())

                labels[i] = y_target.item()
                predictions[i] = anomaly_score.mean().item()
                i += 1

            # try:
            eval_rocauc = roc_auc_score(labels, predictions)
            eval_p_rocauc = roc_auc_score(labels, predictions, max_fpr=0.1)
            # except ValueError:
            #     print("Couldn't compute eval_rocauc/eval_p_rocauc for {}_{}".format(ds.subset_path, ds.machine_id))
            #     eval_rocauc = None
            #     eval_p_rocauc = None

            rocauc_name = "{}_{}_rocauc".format(ds.subset_path, ds.machine_id)
            p_rocauc_name = "{}_{}_p_rocauc".format(ds.subset_path, ds.machine_id)

            print(rocauc_name, eval_rocauc)
            print(p_rocauc_name, eval_p_rocauc)

            self.experiment.log_scalar(rocauc_name, eval_rocauc)
            self.experiment.log_scalar(p_rocauc_name, eval_p_rocauc)

            # save evaluation results as artifact
            if(save_eval_results):
                print("Saving evaluation results as artifact.")
                tmp_filename = ""
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    pickle.dump(eval_result, open(tmp_file.name, 'wb'))
                    tmp_filename = tmp_file.name

                self.experiment.add_artifact(tmp_filename, name="{}_{}_evaluation_results".format(ds.subset_path, ds.machine_id))
                os.remove(tmp_filename)


class AutoEncoderTrainer(BaseTrainer):

    def train_(self, epoch):
        """Train model using training set."""
        running_train_loss = 0.0
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            x, _, y = prepare_batch(batch, self.device)
            y = adapt_dimension(y, self.target_mode)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            # print(outputs)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            running_train_loss += loss.item()
            # if i % 10 == 0:
            #     print("Finished processing batch {}".format(i))

        return running_train_loss / len(self.train_loader)

    def validate_(self, epoch):
        running_valid_loss = 0.0
        for batch in self.validation_loader:
            x, _, y = prepare_batch(batch, self.device)
            y = adapt_dimension(y, self.target_mode)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            running_valid_loss += loss.item()

        return running_valid_loss / len(self.validation_loader)


class FlowTrainer(BaseTrainer):
    def train_(self, epoch):
        running_train_loss = 0.0
        self.model.train()

        print("# batches:", len(self.train_loader))
        for i, batch in enumerate(self.train_loader):
            x, _, y = prepare_batch(batch, self.device)
            y = adapt_dimension(y, self.target_mode)
            if self.transpose_flatten:
                x = adapt_dimension(x, "transpose_flatten")
            else:
                x = adapt_dimension(x, "flatten")
            loss = -self.model.log_prob(x, y).mean(0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_train_loss += loss.item()

        return running_train_loss / len(self.train_loader)


    @torch.no_grad()
    def validate_(self, epoch):
        self.model.eval()
        cond_label_size = self.cond_label_size
        logprior = torch.tensor(1 / cond_label_size).log().to(self.device)
        loglike = [[] for _ in range(cond_label_size)]

        for i in range(cond_label_size):
            # make one-hot labels
            labels = torch.zeros(self.validation_loader.batch_size, cond_label_size).to(self.device)
            labels[:, i] = 1

            for batch in self.validation_loader:
                # x, _, _ = prepare_batch(batch, self.device)
                x, _, _ = batch
                if self.transpose_flatten:
                    x = adapt_dimension(x, "transpose_flatten")
                else:
                    x = adapt_dimension(x, "flatten")
                x = x.to(self.device).float()
                loglike[i].append(self.model.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)
        print("samples", len(self.validation_loader.dataset))
        print("batches", len(self.validation_loader))
        print("logprobs", logprobs.shape)
        logprob_mean = logprobs.mean(0)
        # logprob_std = 2 * logprobs.var(0).sqrt() / math.sqrt(len(self.validation_loader.dataset))

        return -logprob_mean.item()

