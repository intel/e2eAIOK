import torch
from peft import LoraModel
from peft.import_utils import is_bnb_available, is_bnb_4bit_available
from peft.tuners.lora import _get_submodules, mark_only_lora_as_trainable, _freeze_adapter, LoraLayer, Embedding, Conv2d, Linear, Conv1D

if is_bnb_available():
    import bitsandbytes as bnb
    from peft.tuners.lora import Linear8bitLt
    if is_bnb_4bit_available():
        from peft.tuners.lora import Linear4bit

class DeltaLoraSearchSpace:
    @classmethod
    def generate_search_space(cls, model, denas_config,
                              max_r=12, r_step=1, 
                              max_alpha=6, alpha_step=1):
        search_space = {}
        search_space_name = ["num_hidden_layers", "r", "alpha"]
        for i in range(getattr(model.config, denas_config.layer_name)):
            search_space["num_hidden_layers_{}".format(i)] = [0,1]
            search_space["r_{}".format(i)] = list(range(r_step, max_r+r_step, r_step))
            search_space["alpha_{}".format(i)] = list(range(alpha_step, max_alpha+alpha_step, alpha_step))
        return search_space, search_space_name
    
    @classmethod
    def generate_supernet(cls, model, denas_config, search_space):
        supernet_config = {}
        supernet_config["num_hidden_layers"] = [search_space["num_hidden_layers_{}".format(i)][-1] for i in range(getattr(model.config, denas_config.layer_name))]
        supernet_config["r"] = [search_space["r_{}".format(i)][-1] for i in range(getattr(model.config, denas_config.layer_name))]
        supernet_config["alpha"] = [search_space["alpha_{}".format(i)][-1] for i in range(getattr(model.config, denas_config.layer_name))]
        return supernet_config

class DeltaLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name, model_param):
        super().__init__(model, config, adapter_name)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.model_param = model_param
        self.adapter_name = adapter_name
        self.set_sample_config(self.model_param)

    def _create_org_module(self, target):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, Linear8bitLt):
            eightbit_kwargs = {}
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = bnb.nn.Linear8bitLt(input_features=target.in_features, output_features=target.out_features, bias=bias, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target, Linear4bit):
            fourbit_kwargs = {}
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = bnb.nn.Linear4bit(target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, Embedding):
            new_module = torch.nn.Embedding(num_embeddings=target.num_embeddings, embedding_dim=target.embedding_dim)
        elif isinstance(target, Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            if isinstance(target, Linear):
                in_features, out_features = target.in_features, target.out_features
            elif isinstance(target, Conv1D):
                in_features, out_features = (target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape)
            elif isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
            else:
                raise ValueError(
                    f"Target module {target} is not supported. ")
            new_module = torch.nn.Linear(in_features, out_features, bias=bias)
        return new_module

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if old_module.weight.shape != new_module.weight.shape:
            old_module.weight.data = old_module.weight.data.T
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def set_sample_config(self, model_config):
        lora_config = self.peft_config[self.adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            l_num = [int(k) for k in key.split(".") if k.isnumeric()].pop()
            r = model_config["r"][l_num]
            alpha = model_config["alpha"][l_num] * r
            if model_config["num_hidden_layers"][l_num]:
                lora_config.r = r
                lora_config.lora_alpha = alpha
                if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                    target.update_layer_conv2d(self.adapter_name, lora_config.r, lora_config.lora_alpha, lora_config.lora_dropout, lora_config.init_lora_weights,)
                elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                    target.update_layer_embedding(self.adapter_name, lora_config.r, lora_config.lora_alpha, lora_config.lora_dropout, lora_config.init_lora_weights,)
                elif isinstance(target, LoraLayer):
                    target.update_layer(self.adapter_name, lora_config.r, lora_config.lora_alpha, lora_config.lora_dropout, lora_config.init_lora_weights,)
                else:
                    new_module = self._create_new_module(lora_config, self.adapter_name, target)
                    self._replace_module(parent, target_name, new_module, target)
            else:
                new_module = self._create_org_module(target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again.")
        mark_only_lora_as_trainable(self.model, self.peft_config[self.adapter_name].bias)
        if self.peft_config[self.adapter_name].inference_mode:
            _freeze_adapter(self.model, self.adapter_name)
        return self.model
