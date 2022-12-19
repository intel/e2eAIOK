import logging

class BasicFinetunner:
    ''' Basic finetunner

    '''
    def __init__(self,pretrained_network,is_frozen=True):
        ''' init method

        :param pretrained_network: the pretrained_network
        :param is_frozen: whether frozen the finetunned bottom layers
        '''
        self._pretrained_params = pretrained_network.state_dict()
        self.is_frozen = is_frozen

    def __str__(self):
        _str = "BasicFinetunner:\n\t"
        _str += "is_frozen:%s\n\t" % self.is_frozen
        return  _str

    def finetune_network(self,target_network):
        ''' finetune target network

        :param target_network: the target network
        :return:
        '''
        named_parameters = {name: parameter for (name, parameter) in target_network.named_parameters()}
        ############################## finetune #######################
        self.finetuned_state_keys = []
        for key in target_network.state_dict():
            if 'num_batches_tracked' in key:
                continue
            target_v = target_network.state_dict()[key]
            if key in self._pretrained_params:
                pretrained_v = self._pretrained_params[key]
                if target_v.shape == pretrained_v.shape:
                    target_v.data.copy_(pretrained_v.data)  # Copy the data of parameters
                    self.finetuned_state_keys.append(key)
                    if self.is_frozen and key in named_parameters:
                        named_parameters[key].requires_grad = False
                else:
                    logging.error('could not load layer: %s; mismatch shape: target [%s] != pretrained [%s]'%(
                        key, (target_v.shape), (pretrained_v.shape)))
                    print('could not load layer: %s; mismatch shape: target [%s] != pretrained [%s]'%(
                        key, (target_v.shape), (pretrained_v.shape)))
            else:
                logging.error('could not load layer: %s, not in pretrained'%key)
                print('could not load layer: %s, not in pretrained'%key)
        logging.info("Finetune %s keys"%len(self.finetuned_state_keys))