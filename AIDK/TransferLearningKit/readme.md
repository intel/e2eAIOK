# Transfer Learning Kit

## 1. Usage

The transfer learning kit is easily integrated following the follow steps without any modification on original model:

1. We need to wrap the original model with a distiller and an adapter by `make_transferrable()`:
    ```
    model = ... # orignal model
    model = make_transferrable(model,loss,
                       distiller_feature_size,distiller_feature_layer_name,
                       adapter_feature_size,adapter_feature_layer_name,
                       distiller,adapter,
                       training_dataloader,adaption_source_domain_training_dataset,
                       transfer_strategy,enable_target_training_label)
    # model act as the original one
    ```
    - param model: the backbone model. If model does not have loss method, then use loss argument.
    - param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    - param distiller_feature_size: input feature size of distiller. If no distillation, then could be none.
    - param distiller_feature_layer_name: specify the layer output, which is from model, as input feature of distiller. If no distillation, then could be none.
    - param adapter_feature_size: input feature size of adapter. If no adaption, then could be none.
    - param adapter_feature_layer_name: specify the layer output, which is from model, as input feature of adapter. If no adaption, then could be none.
    - param distiller: a distiller. If no distillation, then could be none.
    - param adapter: an adapter. If no adaption, then could be none.
    - param training_dataloader: training dataloader.
    - param adaption_source_domain_training_dataset: source domain training dataset for adaption. If no adaption, then could be none.
    - param transfer_strategy: transfer strategy.
    - param enable_target_training_label: During training, whether use target training label or not.
    - return: a TransferrableModel
   
    And the strategy could be:
    ```
    OnlyFinetuneStrategy               : pretraining-finetuning, and the pretrained model is the same as the target model
    OnlyDistillationStrategy           : distillation
    OnlyDomainAdaptionStrategy         : domain adaption
    FinetuneAndDomainAdaptionStrategy  : pretraining-finetuning and domain adaption
    DistillationAndAdaptionStrategy    : distillation and domain adaption
    ```
    
    The new generated model act the same role as original model, and both can replace each other.

2. If we need to logging the training loss or training metrics, we suggest to distinguish the new model and the original model, because maybe we want to log more details about transfer learning:
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
   And the training sub-workflow maybe:
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