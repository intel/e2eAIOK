import ast
import torch
from e2eAIOK.DeNas.cv.benchmark_network_latency import get_model_latency
from e2eAIOK.DeNas.nlp.utils import get_bert_latency
from e2eAIOK.DeNas.thirdparty.utils import get_hf_latency

NETWORK_LATENCY = {"cnn": get_model_latency,
                    "bert": get_bert_latency,
                    "transformer": get_model_latency,
                    "hf": get_hf_latency}

def decode_arch_tuple(arch_tuple):
    arch_tuple = ast.literal_eval(arch_tuple)
    depth = int(arch_tuple[0])
    mlp_ratio = [float(x) for x in (arch_tuple[1:depth+1])]
    num_heads = [int(x) for x in (arch_tuple[depth + 1: 2 * depth + 1])]
    embed_dim = int(arch_tuple[-1])
    return depth, mlp_ratio, num_heads, embed_dim

def get_total_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_pruned_parameters_count(pruned_model):
    params = 0
    for param in pruned_model.parameters():
        if param is not None and param.requires_grad:
            params += torch.nonzero(param).size(0)
    return params
