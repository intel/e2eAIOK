# Transfer Learning Kit

## 1. Usage

The transfer learning kit is easily integrated following the follow steps:

1. The original model should provide the follow properties: loss function, distiller input, distiller input size, adapter input, adapter input size, without any structure modification on original model.
  ```
  class ABCNet(nn.Module):
        ...
        def __init__(self,num_classes):
            ...
            self.distiller_input       # distiller input, which is the intermediate component output of the model
            self.distiller_input_size  # distiller input size
            self.adapter_input         # adapter input, which is the intermediate component output of the model
            self.adapter_input_size    # adapter input size
        ...
        def loss(self,output, label):
        ''' loss function

        :param output: model prediction
        :param label: ground truth
        :return: loss function
        '''  
  ```
2. To transfer knowledge from a target-similar dataset to target task, we need to provide 2 datasets: the target-similar dataset(called `source dataset`), the target dataset. To be compatible with original workflow, 
We provide a wrapper class `ComposedDataset` which composes target dataset and source dataset into a composed dataset. 
  ```
  from dataset.composed_dataset import ComposedDataset
  target_train_dataset = ...
  target_validation_dataset = ...
  target_test_dataset = ...
  
  source_train_dataset = ...
  composed_train_dataset = ComposedDataset(target_train_dataset,source_train_dataset)
  # now composed_train_dataset act the same role as target_train_dataset
  ```
3. We need to wrap the original model with a distiller and an adapter by `make_transferrable()`:
   - We can transfer knowledge from a pretrained model and finetune on our own task. We can achieve this with one line modified:
     ```commandline
     from engine_core.transferrable_model import make_transferrable,TransferStrategy
     # ...
     model = make_transferrable(model,adapter=None,distiller=None,transfer_strategy=TransferStrategy.OnlyFinetuneStrategy,
     enable_target_training_label=True)
     # ...
     ```
   - We can transfer knowledge from a pretrained model and distill it. We can achieve this by two lines modified:
     ```commandline
     from engine_core.transferrable_model import make_transferrable,TransferStrategy
     from engine_core.distiller.basic_distiller import BasicDistiller
     # ...
     distiller = BasicDistiller(...)
     model = make_transferrable(model,adapter=None,distiller=distiller,transfer_strategy=TransferStrategy.OnlyDistillationStrategy,
     enable_target_training_label=True)
     # ...
     ```
   - We can transfer knowledge from a target-similar dataset to target task. We can achieve this by two lines modified:
     ```commandline
     from engine_core.transferrable_model import make_transferrable,TransferStrategy
     from engine_core.adapter.factory import createAdapter
     # ...
     adapter = createAdapter('CDAN',...)
     model = make_transferrable(model,adapter=adapter,distiller=None,transfer_strategy=TransferStrategy.OnlyDomainAdaptionStrategy,
     enable_target_training_label=True)
     # ...
     ```
   - We can both transfer knowledge from a pretrained model and target-similar dataset to target task. We can achieve this by three lines modified:
     ```commandline
     from engine_core.transferrable_model import make_transferrable,TransferStrategy
     from engine_core.distiller.basic_distiller import BasicDistiller
     from engine_core.adapter.factory import createAdapter
     # ...
     distiller = BasicDistiller(...)
     adapter = createAdapter('CDAN',...)
     model = make_transferrable(model,adapter=adapter,distiller=distiller,transfer_strategy=TransferStrategy.DistillationAndAdaptionStrategy,
     enable_target_training_label=True)
     # ...
     ```
    The new generated model act the same role as original model, and both can replace each other.

4. There would exist a conflict if we provide a composed dataset and an original model. So the training iteration should be adjusted by:
  ```
 for (cur_step,(data, label)) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if isinstance(train_dataloader.dataset, ComposedDataset) and (not isinstance(model, TransferrableModel)):
            output = model(data[0])
            loss_value = model.loss(output, label[0])
        else:
            output = model(data)
            loss_value = model.loss(output, label)
        loss_value.backward()
  ```

5. If we need to logging the training loss or training metrics, we suggest to distinguish the new model and the original model, because maybe we want to log more details about transfer learning:
  ```
  for (cur_step,(data, label)) in enumerate(train_dataloader):
     ...
     if cur_step % logging_interval == 0:
          if isinstance(model,TransferrableModel): # new model metrics
               metric_values = model.get_training_metrics(output,label,loss_value,metric_fn_map)
          else: # original model metrics
               ...
  ```

## 2. Project Structure

```
src/
   dataset/ -------------------------------------------------- built-in datasets and helper function 
           composed_dataset.py ------------------------------- compose multi domain datasets
           image_list.py ------------------------------------- dataset for MNIST
           office31.py --------------------------------------- dataset for office31
   engine_core/ ---------------------------------------------- transfer learning engine
              adapter/ --------------------------------------- domain adapters
                     adversarial/ ---------------------------- adversarial domain adapters
                                 adversarial_adapter.py ------ base adversarial domain adapter
                                 cdan_adapter.py ------------- cdan domain adapter
                                 dann_adapter.py ------------- dann domain adapter
                                 grl.py ---------------------- Gradient Reverse Layer
                     factory.py ------------------------------ domain adapter factory  
              backbone/ -------------------------------------- backbone networks
                      factory.py ----------------------------- backbone factory
                      lenet.py ------------------------------- built-in lenet network
                      resnet.py ------------------------------ built-in resnet network
              distiller/ ------------------------------------- knowledge distillers
                      basic_distiller.py --------------------- basic knowledge distiller
              transfeerable_model.py ------------------------- the core interface to make a model transferrable
   training/ ------------------------------------------------- training helper
              metrics.py ------------------------------------- evaluation metrics
              train.py --------------------------------------- Trainer and Evaluator for training and evaluating
              utils.py --------------------------------------- helper function for training
   main.py --------------------------------------------------- the main file                         
```

## 3. Project Components

1. **Backbone Factory** : creates a backbone net according to predefined backbone or user-provided backbone to make basic prediction 
2. **Domain Adaptor** : creates a domain adaption net (called “adaptor”) to transfer knowledge from source domain to target domain 
3. **Knowledge Distiller** : creates a knowledge distillation net (called “distiller”) to transfer knowledge from pretrained model to target model 
4. **Transferrable Model** : creates a customized and transferrable model which is a wrapper of backbone, according to the backbone, adaptor and distiller 

<img src="./doc/imgs/components.png" width="700px" />

## 4. WorkFlow

Our workflow is the Traditional Workflow with Transfer Learning Kit integrated.

1. Traditional Workflow: It is up to the user. For example:
   ```commandline
    create  target dataset
    create  model
    create trainer
    train training data
    evaluate validate data
    create evaluator
    evaluate test data
   ```
   And the training workflow maybe:
   ```commandline
    iterate dataloader
    output = model(data)
    loss = model.loss(output, label) 
    loss.backward()
    add tensorboard metric
   ```
   For convenience of discussion, we omit some details here, for example, the creation of optimizer.

2. Transfer Learning Kit: It is an effective and convenient tool, which is used to asign Traditional Workflow the power of transfer learning by replacing some components , for example, the dataset and the model. 

<img src="./doc/imgs/workflow.png" width="700px" />