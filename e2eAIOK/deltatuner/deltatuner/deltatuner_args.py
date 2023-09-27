from typing import List
from dataclasses import dataclass, field
from transformers import AutoTokenizer

@dataclass
class DeltaTunerArguments:
    search_engine: str = field(default="EvolutionarySearchEngine", metadata={"help": "Which search strategy used in the DE-NAS. Can be 'EvolutionarySearchEngine', 'RandomSearchEngine' and 'SigoptSearchEngine'"})
    model_id: str = field(default=None, metadata={"help": "The model id in the hugging face or the local model path"})
    batch_size: int = field(default=4, metadata={"help": "The batch size used in the search process to calculate the inference"})
    max_epochs: int = field(default=10, metadata={"help": "The max epochs of the search processing"})
    scale_factor: int = field(default=10, metadata={"help": "The scaling factor used in the EA search"})
    select_num: int = field(default=50, metadata={"help": "The num of top candidates stored in the EA search process"})
    population_num: int = field(default=50, metadata={"help": "The num of population candidates used in the EA/Random search process"})
    m_prob: float = field(default=0.2, metadata={"help": "The m probability to produce new mutation network depth"})
    s_prob: float = field(default=0.4, metadata={"help": "The s probability to produce new other mutation network structures"})
    crossover_num: int = field(default=25, metadata={"help": "The number of crossover in the EA search process"})
    mutation_num: int = field(default=25, metadata={"help": "The number of mutation in the EA search process"})
    img_size: int = field(default=32, metadata={"help": "The input or output sequence length of the model."})
    expressivity_weight: float = field(default=0, metadata={"help": "The weight of expressivity score in the de-score"})
    complexity_weight: float = field(default=0, metadata={"help": "The weight of complexity score in the de-score"})
    diversity_weight: float = field(default=1, metadata={"help": "The weight of diversity score in the de-score"})
    saliency_weight: float = field(default=1, metadata={"help":"The weight of saliency score in the de-score"})
    latency_weight: float = field(default=0, metadata={"help":"The weight of latency score in the de-score"})
    max_param_limits: int = field(default=None, metadata={"help": "The max M number of trainable parameter size"})
    min_param_limits: int = field(default=None, metadata={"help": "The min M number of trainable parameter size"})
    budget_latency_max: int = field(default=None, metadata={"help": "The max latency limit of the candidate network (etc., '3*10**6' second for your estimation of max latency of LLM forward)"})
    budget_latency_min: int = field(default=None, metadata={"help": "The min latency limit of the candidate network (etc., '1*10**6' second for your estimation of min latency of LLM forward)"})
    layer_name: str = field(default="num_hidden_layers", metadata={"help": "The layer name for a specific model search space"})
    best_model_structure: str = field(default=None, metadata={"help": "The file path to load the best model structure"})
    search_space_name: List[str] = field(default=None, metadata={"help": "The search space name of delta layers"})
    random_seed: int = field(default=12345, metadata={"help": "The random seed of the deltatuner search process"})
    denas: bool = field(default=True, metadata={"help": "Whether to use the denas"})
    ssf_target_module: List[str] = field(default_factory=lambda: None, metadata={"help": "Target modules for the SSF method."},)
    task_type: str = field(default="CAUSAL_LM", metadata={"help": "The task type of the deltatuner model"})