import torch.nn as nn
from .utils import save_check_logits, load_logits

class BasicDistiller(nn.Module):
    ''' BasicDistiller

    '''
    def __init__(self,pretrained_model, is_frozen=True, use_saved_logits=False, topk=0, num_classes=10, teacher_type=None):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param is_frozen: whether frozen teacher when training
        :param use_saved_logits: whether train with pre-saved logits
        :param topk: if use logits, save top k logits, 0 means save all logits
        :param num_classes: num of classification classes
        :param teacher_type: teacher model type
        '''
        super(BasicDistiller, self).__init__()
        self.pretrained_model = pretrained_model
        self.is_frozen = is_frozen
        self.use_saved_logits = use_saved_logits
        self.topk = topk
        self.num_classes = num_classes
        self.pretrained_model_type = teacher_type
        
        if self.is_frozen:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def prepare_logits(self, dataloader, epochs, start_epoch=0, device="cpu", save_flag=True, check_flag=False):
        ''' Save logits function.

        :param dataloader: the dataloader
        :param epochs: totally saved epochs
        :param start_epoch: the starting epoch to save
        :param device: whether use "cpu" or "cpu"
        :param save_flag: whether save logits
        :param check_flag: whether check logits
        '''
        save_check_logits(self.pretrained_model, dataloader, epochs, 
                        start_epoch=start_epoch, topk=self.topk, num_classes=self.num_classes, model_type=self.pretrained_model_type, 
                        device=device, save_flag=save_flag, check_flag=check_flag)
    
    def forward(self, x):
        ''' forward function.

        :param x: the input
        '''
        if not self.use_saved_logits:
            output = self.pretrained_model(x)
            output = output.logits if self.pretrained_model_type is not None and self.pretrained_model_type.startswith("huggingface") else output
            return output
        else:
            if not isinstance(x, list) or len(x)!=2:
                raise RuntimeError("need saved logits for distiller")
            output = load_logits(x[1],topk=self.topk,num_classes=self.num_classes)
            return output

    def loss(self,teacher_logits, student_logits,**kwargs):
        ''' Loss function.

        :param teacher_logits: the teacher logits
        :param student_logits: the student logits
        '''
        distiller_loss = nn.MSELoss()(teacher_logits,student_logits)
        return distiller_loss

