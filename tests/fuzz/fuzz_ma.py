import atheris
import sys
import timm
from e2eAIOK.ModelAdapter.engine_core.finetunner import BasicFinetunner
from e2eAIOK.ModelAdapter.engine_core.transferrable_model import *


@atheris.instrument_func
def fuzz_func(data):
  backbone = timm.create_model('resnet50', pretrained=False, num_classes=100)
  pretrained_model = timm.create_model('resnet18', pretrained=False, num_classes=11221)
  finetuner = BasicFinetunner(pretrained_model, is_frozen=False)
  loss_fn = torch.nn.CrossEntropyLoss()
  model = make_transferrable_with_finetune(backbone, loss_fn, finetuner)

if __name__ == "__main__":
  atheris.Setup(sys.argv, fuzz_func)
  atheris.Fuzz()