import torch
import torch.nn as nn
import torch.nn.functional as F

class KD(nn.Module):
    ''' KD Distiller

    '''
    def __init__(self, pretrained_model, is_frozen, temperature ):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param is_frozen: whether frozen teacher when training
        '''
        super(KD, self).__init__()
        self.pretrained_model = pretrained_model
        self._is_frozen = is_frozen
        self.temperature = temperature
        if is_frozen:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.pretrained_model(x)

    def kd_loss(self, logits_student, logits_teacher, temperature):
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature**2
        return loss_kd

    def loss(self,teacher_logits, student_logits, **kwargs):
        distiller_loss = self.kd_loss(student_logits, teacher_logits, self.temperature)
        return distiller_loss