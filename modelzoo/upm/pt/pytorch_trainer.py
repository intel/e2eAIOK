import logging
import os
import time
import torch
import pt.extend_distributed as ext_dist
import sklearn.metrics
import numpy as np


class PytorchTrainer:
    def __init__(self, args, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        self.num_epochs = args.num_epochs
        self.metric_threshold = args.metric_threshold

    def train(self, train_dataset, eval_dataset, metrics_print_interval=10):
        logger = logging.getLogger('upm')
        if ext_dist.my_size > 1:
            self.model = ext_dist.DDP(self.model)
        
        for epoch in range(0, self.num_epochs):
            for step, (x, y) in enumerate(train_dataset):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 10 == 0:
                    print(f'step: {step}, loss: {loss}')
            scores = []
            targets = []
            for (x, y) in eval_dataset:
                test = self.model(x)
                test = ext_dist.all_gather(test, None)
                y = ext_dist.all_gather(y, None)
                
                test = test.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                scores.append(test)
                targets.append(y)
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)
            metrics = sklearn.metrics.roc_auc_score(targets, scores)
            print(f'metric: {metrics}')
            if metrics >= self.metric_threshold:
                logger.info(f'early stop at {metrics}')
                break
        return metrics