import ast
import os
import logging
import torch

from trainer.ModelBuilder import BaseModelBuilder
from nlp.supernet_bert import SuperBertForQuestionAnswering, BertConfig
from nlp.utils import get_subconfig

class BertModelBuilder(BaseModelBuilder):
    def __init__(self, args):
        self.args = args
        self.model_dir = self.args.model_dir
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def decode_arch(self):
        subbert_config = None
        base_arches_file = self.args.arches_file
        with open(os.path.join(base_arches_file, "best_model_structure.txt"), 'r') as fin:
            for line in fin:
                line = line.strip()
                subbert_config = get_subconfig(ast.literal_eval(line))
        self.logger.info('subbert_config: {}'.format(subbert_config))
        return subbert_config

    def init_model(self):
        config = BertConfig.from_pretrained(self.model_dir, num_labels=2, finetuning_task='sst-2')
        if self.args.data_set == "SQuADv1.1":
            config.num_labels = 2
            config.finetuning_tasl = "squad1"
        model = SuperBertForQuestionAnswering.from_pretrained(self.model_dir, config)
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        model.to(device)
        return model

    def create_model(self, ext_dist):
        model = self.init_model()
        if ext_dist.my_size > 1:
            model = ext_dist.DDP(model, find_unused_parameters=True)
        subbert_config = self.decode_arch()
        model.module.set_sample_config(subbert_config) if hasattr(model, 'module') \
            else model.set_sample_config(subbert_config)
        size = model.module.calc_sampled_param_num() if hasattr(model, 'module') \
            else model.calc_sampled_param_num()
        self.logger.info('Total parameters: {}'.format(size))
        return model, subbert_config