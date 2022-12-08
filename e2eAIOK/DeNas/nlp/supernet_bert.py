from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
import os
import json
import copy
import tarfile
import tempfile


import torch
from torch import embedding, nn
from torch.nn import CrossEntropyLoss

from module.nlp.Linear_super import LinearSuper as SuperLinear
from module.nlp.layernorm_super import LayerNormSuper as SuperBertLayerNorm
from module.nlp.bert_embedding_super import SuperBertEmbeddings
from module.nlp.bert_encoder_super import SuperBertEncoder
from module.nlp.bert_pooler_super import SuperBertPooler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"


class BertConfig(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path

        # Load config
        config = cls.from_json_file(config_file)

        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BertPreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SuperBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        if not os.path.exists(resolved_config_file):
            resolved_config_file = os.path.join(pretrained_model_name_or_path, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        model = cls(*inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = archive_file
            cache_dir=cache_dir
        except EnvironmentError:
            logger.error(
                "Model name was not found in model name list. "
                "We assumed 'model_name' was a path or url but couldn't find any file "
                "associated to this path or url.")
            return None
        if resolved_archive_file == archive_file:
            print("loading archive file {}".format(archive_file))
        else:
            print("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            print("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(*inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None

            if 'bert' not in key:
                new_key = 'bert.' + key
            if new_key:
                if 'embedding' in key and 'LayerNorm' not in key:
                    tmp = new_key.split('.')
                    new_key = '.'.join(tmp[:-1]) + '.embedding.' + tmp[-1]
                if 'layer' in key:
                    new_key = new_key.replace('layer', 'layers')
            else:
                if 'embedding' in key and 'LayerNorm' not in key:
                    tmp = key.split('.')
                    new_key = '.'.join(tmp[:-1]) + '.embedding.' + tmp[-1]
                if 'layer' in key:
                    new_key = key.replace('layer', 'layers')

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        #print("!!!!!model parameters!!!!!")
        #for param_tensor in model.state_dict():
        #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #print("!!!!!saved parameters!!!!!")
        #for param_tensor in state_dict:
        #    print(param_tensor, "\t", state_dict[param_tensor].size()) 
        #sys.exit()
        
        load(model, prefix=start_prefix)
        #missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class SuperBertModel(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(SuperBertModel, self).__init__(config)
        self.embeddings = SuperBertEmbeddings(config)
        self.encoder = SuperBertEncoder(config)
        self.pooler = SuperBertPooler(config)
        self.dense_fit = SuperLinear(config.hidden_size, fit_size)

        self.hidden_size = config.hidden_size
        self.qkv_size = self.hidden_size

        try:
            self.qkv_size = config.qkv_size
        except:
            self.qkv_size = config.hidden_size

        self.fit_size = fit_size
        self.head_number = config.num_attention_heads
        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config):
        self.embeddings.set_sample_config(subbert_config['sample_hidden_size'])
        self.encoder.set_sample_config(subbert_config)
        self.pooler.set_sample_config(subbert_config['sample_hidden_size'])

    def calc_sampled_param_num(self):
        emb_numel = self.embeddings.calc_sampled_param_num()
        encoder_numel = self.encoder.calc_sampled_param_num()
        pooler_numel = self.pooler.calc_sampled_param_num()

        #logger.info('===========================')
        #logger.info('emb_numel: {}\n'.format(emb_numel))
        #logger.info('encoder_numel: {}\n'.format(encoder_numel))
        #logger.info('pooler_numel: {}\n'.format(pooler_numel))
        #logger.info('all parameters: {}\n'.format(emb_numel + encoder_numel + pooler_numel))
        #logger.info('===========================')
        return emb_numel + encoder_numel + pooler_numel

    def forward(self, input_ids, 
                attention_mask=None, token_type_ids=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(input_ids,
                                           token_type_ids=token_type_ids)

        
        all_encoder_layers, all_encoder_att = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers[-1], pooled_output


class SuperBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(SuperBertForQuestionAnswering, self).__init__(config)
        self.bert = SuperBertModel(config)
        self.qa_outputs = SuperLinear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config):
        self.bert.set_sample_config(subbert_config)
        self.qa_outputs.set_sample_config(subbert_config['sample_hidden_size'], 2)

    def calc_sampled_param_num(self):
        return self.bert.calc_sampled_param_num()

    def save_pretrained(self, save_directory):

        assert os.path.isdir(save_directory), "Saving path should be a directory where " \
                                              "the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, x):

        input_ids, attention_mask, token_type_ids = x.split(1, -1)
        input_ids = input_ids.squeeze(-1)
        attention_mask = attention_mask.squeeze(-1)
        token_type_ids = token_type_ids.squeeze(-1)
        encoded_layers, pooled_output = self.bert(input_ids, 
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        last_sequence_output = encoded_layers
        
        logits = self.qa_outputs(last_sequence_output)
        #start_logits, end_logits = logits.split(1, dim=-1)
        #start_logits = start_logits.squeeze(-1)
        #end_logits = end_logits.squeeze(-1)

        #logits = torch.cat((start_logits, end_logits), -1)

        return logits


class CrossEntropyQALoss(nn.Module):
    def __init__(self, ignored_index):
        super(CrossEntropyQALoss, self).__init__()
        self.ignored_index = ignored_index
        self.loss = CrossEntropyLoss(ignore_index=ignored_index)

    def forward(self, output, target):
        target_s, target_e = torch.split(target, int(target.size()[-1]/2), -1)
        output_s, output_e = torch.split(output, int(output.size()[-1]/2), -1)
        if len(target_s.size()) > 1:
            target_s = target_s.squeeze(-1)
        if len(target_e.size()) > 1:
            target_e = target_e.squeeze(-1)
        if len(output_s.size()) > 1:
            output_s = output_s.squeeze(-1)
        if len(output_e.size()) > 1:
            output_e = output_e.squeeze(-1)
        target_s.clamp_(0, self.ignored_index)
        target_e.clamp_(0, self.ignored_index)
        start_loss = self.loss(output_s, target_s)
        end_loss = self.loss(output_e, target_e)
        cls_loss = (start_loss + end_loss) / 2
        return cls_loss 

