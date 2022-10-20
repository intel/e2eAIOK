import torch
import torch.nn as nn
import torch.nn.functional as F

class KD(nn.Module):
    ''' KD Distiller

    '''
    def __init__(self, pretrained_model, temperature, is_frozen=True, teacher_forward=True, teacher_type=None):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param temperature: the temperature for KD 
        :param is_frozen: whether frozen teacher when training
        :param teacher_forward: whether do teacher forwarding, set False when train with pre-saved logits
        :param teacher_type: teacher model type
        '''
        super(KD, self).__init__()
        self.pretrained_model = pretrained_model
        self.temperature = temperature
        self.is_frozen = is_frozen
        self.teacher_forward = teacher_forward
        self.teacher_type = teacher_type

        if self.is_frozen:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.teacher_forward:
            output = self.pretrained_model(x)
            output = (output.logits,None) if self.teacher_type == "vit_base_224_in21k_ft_cifar100" else output
            return output
        else:
            return None,None

    def kd_loss(self, logits_student, logits_teacher, temperature):
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature**2
        return loss_kd

    def loss(self,teacher_logits, student_logits, **kwargs):
        distiller_loss = self.kd_loss(student_logits, teacher_logits, self.temperature)
        return distiller_loss