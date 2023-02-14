import ast
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