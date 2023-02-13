import torch
from easydict import EasyDict as edict
from e2eAIOK.DeNas.asr.model_builder_denas_asr import ModelBuilderASRDeNas

class TestDeNasASRModelBuilder:

    '''
    Test ModelBuilderASRDeNas.load_pretrained_model()
    '''
    def test_load_pretrained_model(self):
        cfg = {
            "input_shape": [8, 10, 80],
            "num_blocks": 3,
            "num_layers_per_block": 1,
            "out_channels": [64, 64, 64],
            "kernel_sizes": [5, 5, 1],
            "strides": [2, 2, 1],
            "residuals": [False, False, True],
            "input_size": 1280, #n_mels/strides * out_channels
            "d_model": 16,
            "encoder_heads": [4],
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "mlp_ratio": [4.0],
            "d_ffn": 64,
            "transformer_dropout": 0.1,
            "output_neurons": 5000,
            "best_model_structure": None,
            "model": "/tmp/model.pt"
        }
        cfg = edict(cfg)

        # save model
        model1 = ModelBuilderASRDeNas(cfg).create_model()
        torch.save(model1.state_dict(), cfg["model"])
        # load saved model
        model2 = ModelBuilderASRDeNas(cfg).load_pretrained_model()
        
