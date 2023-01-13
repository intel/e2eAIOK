import torch
import json
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig

class SuperHFModel(AutoModel):

    @classmethod
    def set_sample_config(cls, pretrained_model_name_or_path, **kwargs):
        is_pretrained = kwargs.pop("is_pretrained", True)
        # Create the candidate net with random initialization
        candidate_hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        candidate_hf = cls.from_config(candidate_hf_config)

        if is_pretrained:
            # Create the super net
            super_hf = cls.from_pretrained(pretrained_model_name_or_path)
            # Load pre-trained weight from super net
            candidate_hf_state_dict = candidate_hf.state_dict()
            super_hf_state_dict = super_hf.state_dict()

            new_candidate_hf_state_dict = {}
            for k in candidate_hf_state_dict:
                super_hf_state = super_hf_state_dict[k]
                candidate_hf_state = super_hf_state
                for dim, size in enumerate(candidate_hf_state_dict[k].size()):
                    candidate_hf_state = candidate_hf_state.index_select(dim, torch.tensor(range(size)))
                new_candidate_hf_state_dict[k] = candidate_hf_state
            candidate_hf.load_state_dict(new_candidate_hf_state_dict)

        return candidate_hf

    @classmethod
    def search_space_generation(cls, pretrained_model_name_or_path, **kwargs):
        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        search_space = {}
        search_space["num_hidden_layers"] = range(int(hf_config.num_hidden_layers/2), int(hf_config.num_hidden_layers), 1)
        search_space["num_attention_heads"] = range(int(hf_config.num_attention_heads/2), int(hf_config.num_attention_heads), 1)
        search_space["hidden_size"] = range(int(hf_config.hidden_size/2), int(hf_config.hidden_size), 16)
        for k in kwargs:
            search_space[k] = range(int(kwargs[k]/2), int(kwargs[k]), 16)
        return search_space
