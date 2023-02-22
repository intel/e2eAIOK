import torch
from e2eAIOK.DeNas.pruner.PytorchPruner import PytorchPruner
from torchvision.models import resnet50
from copy import deepcopy
import torch.nn.utils.prune as prune
from e2eAIOK.DeNas.utils import get_total_parameters_count, get_pruned_parameters_count

class TestDeNasPytorchPruner:

    def test_get_prune_algo(self):
        pruner = PytorchPruner("l1unstructured")
        assert pruner.algo == prune.l1_unstructured

    '''
    Test Pruner.prune(model)
    '''
    def test_prune(self):
        pruner = PytorchPruner("l1unstructured")
        origin_model = resnet50()
        origin_params_count = get_total_parameters_count(origin_model)
        pruned_model = deepcopy(origin_model)
        pruner.prune(pruned_model, 0.9)
        pruned_params_count = get_pruned_parameters_count(pruned_model)
        pruned_params_count_total = get_total_parameters_count(pruned_model)

        assert origin_params_count == pruned_params_count_total
        assert origin_params_count > pruned_params_count
