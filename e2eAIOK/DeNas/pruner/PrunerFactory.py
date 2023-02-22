from e2eAIOK.DeNas.pruner.PytorchPruner import PytorchPruner

PRUNER_BACKENDS = {
        "pytorch": PytorchPruner,
}

class PrunerFactory(object):

    @staticmethod
    def create_pruner(backend, algo, layer_list, exclude_list):
        try:
            if backend.lower() in PRUNER_BACKENDS:
                return PRUNER_BACKENDS[backend](algo, layer_list, exclude_list)
            else:
                raise RuntimeError(f"Pruner backend {backend} is not supported")
        except Exception as e:
            raise RuntimeError(f"Initialize pruner with backend {backend} failed. Error Msg: {e}")