import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_distiller import BasicDistiller

class KD(BasicDistiller):
    ''' KD Distiller

    '''
    def __init__(self, pretrained_model, temperature=4.0, is_frozen=True, use_saved_logits=False, topk=0, num_classes=10, teacher_type=None):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param temperature: the temperature for KD 
        :param is_frozen: whether frozen teacher when training
        :param use_saved_logits: whether train with pre-saved logits
        :param topk: if use logits, save top k logits, 0 means save all logits
        :param num_classes: num of classification classes
        :param teacher_type: teacher model type
        '''
        super(KD, self).__init__(pretrained_model, is_frozen, use_saved_logits, topk, num_classes, teacher_type)
        self.temperature = temperature

    def kd_loss(self, logits_student, logits_teacher, temperature):
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature**2
        return loss_kd

    def loss(self,teacher_logits, student_logits, **kwargs):
        ''' Loss function.

        :param teacher_logits: the teacher logits
        :param student_logits: the student logits
        '''
        distiller_loss = self.kd_loss(student_logits, teacher_logits, self.temperature)
        return distiller_loss