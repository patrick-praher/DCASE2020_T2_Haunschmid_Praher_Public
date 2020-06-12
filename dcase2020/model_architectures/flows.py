import torch.nn as nn
from dcase2020.model_architectures.maf import MAF, MADE, MADEMOG, MAFMOG, RealNVP
from dcase2020.model_architectures.anomaly_score_model import AnomalyScoreModel
import torch

class FlowMAFModel(AnomalyScoreModel):

    def __init__(self, flow_model_type, n_blocks, input_size, hidden_size, n_hidden, cond_label_size, input_order,
                 use_batch_norm, activation_fn='relu'):

        super().__init__()
        if flow_model_type == 'made':
            model = MADE(input_size, hidden_size, n_hidden, cond_label_size,
                         activation_fn, input_order)
        elif flow_model_type == 'mademog':
            # assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
            # model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
            #                 args.activation_fn, args.input_order)
            raise NotImplementedError()
        elif flow_model_type == 'maf':
            model = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size,
                        activation_fn, input_order, batch_norm=use_batch_norm)
        elif flow_model_type == 'mafmog':
            # assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
            # model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden,
            #                args.cond_label_size,
            #                args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
            raise NotImplementedError()
        elif flow_model_type == 'realnvp':
            model = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size,
                            batch_norm=use_batch_norm)

        self.model = model

    def forward(self, x):
        return self.model.forward(x)

    def log_prob(self, x, y):
        y = y.squeeze()
        return self.model.log_prob(x, y)

    def anomaly_score(self, x, y, criterion, device="cuda:0"):
        # print("x", x.shape)
        # print("y", y.shape)
        cond_label_size = y.shape[2]
        # print("Determined cond_label_size", cond_label_size)

        logprior = torch.tensor(1 / cond_label_size).log().to(device)
        loglike = [[] for _ in range(cond_label_size)]

        # x = x.view(x.shape[0], -1)
        for i in range(cond_label_size):
            # make one-hot labels
            labels = torch.zeros(x.size(0), cond_label_size).to(device)
            labels[:, i] = 1

            loglike[i].append(self.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)  # cat all data along label dim
        logprobs = logprior + loglike.logsumexp(dim=1)
        return -logprobs