from e2eAIOK.common.trainer.torch_trainer import TorchTrainer
class CVTrainer(TorchTrainer):
    def __init__(self, cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric):
        super().__init__(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)