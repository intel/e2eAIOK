import torch.nn.utils.prune as prune
import functools


class PytorchPruner():
    def __init__(self, algo_name, layer_list=[], exclude_list=[], **kargs):
        self.layer_list = layer_list
        self.exclude_list = exclude_list
        self.kargs = kargs
        self.algo_name = algo_name
        self.algo = self.get_prune_algo(algo_name)

    def get_prune_algo(self, algo):
        """
            get pytorch pruning algorithm based on algo name
        """
        if algo.lower() == "l1unstructured":
            return prune.l1_unstructured
        elif algo.lower() == "randomunstructured":
            return prune.random_unstructured
        elif algo.lower() == "lnstructured":
            return functools.partial(prune.ln_structured, n=2, dim=0)
        elif algo.lower() == "randomstructured":
            return functools.partial(prune.random_structured, dim=0)
        elif algo.lower() == "globalrandomunstructured":
            return functools.partial(prune.global_unstructured, pruning_method=prune.RandomUnstructured)
        elif algo.lower() == "globall1unstructured":
            return functools.partial(prune.global_unstructured, pruning_method=prune.L1Unstructured)
        else:
            raise RuntimeError(f"Pruning algorithm {algo} is not supported yet")
    
    def prune(self, model, sparsity):
        """
            prune model with the given target sparsity
        """
        mask = {}
        if self.algo_name.startswith("global"):
            params_to_prune = tuple([(layer, 'weight') for name, layer in model.named_modules() if hasattr(layer, 'weight') and name not in self.exclude_list])
            self.algo(params_to_prune, amount=sparsity)
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and name not in self.exclude_list:
                    prune.remove(module, 'weight')
                    mask[name] = module.named_buffers()
        else:
            for name, module in model.named_modules():
                if self.layer_list is None or name in self.layer_list and name not in self.exclude_list:
                    if hasattr(module, 'weight'): 
                        if self.algo_name.endswith("unstructured") or (self.algo_name.endswith("structured") and len(module.weight.shape) > 1):
                            self.algo(module, name='weight', amount=sparsity)
                            mask[name] = module.named_buffers()
                            prune.remove(module, 'weight')
        return model, mask