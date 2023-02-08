import torch.nn.utils.prune as prune


class Pruner():
    def __init__(self, algo, sparsity):
        self.algo = self.get_prune_algo(algo)
        self.sparsity = sparsity

    def get_prune_algo(self, algo):
        if algo.lower() == "l1unstructured":
            return prune.L1Unstructured
        elif algo.lower() == "randomunstructured":
            return prune.RandomUnstructured
        else:
            raise RuntimeError(f"Pruning algorithm {algo} is not supported yet")
    
    def prune(self, model):
        params_to_prune = tuple([(layer, "weight") for layer in model.modules() if hasattr(layer, 'weight')])
        prune.global_unstructured(params_to_prune, self.algo, amount=self.sparsity)
        [prune.remove(module, 'weight') for module in model.modules() if hasattr(module, 'weight')]
        return model