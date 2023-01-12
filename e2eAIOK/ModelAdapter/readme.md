# AIOK Model Adaptor
## INTRODUCTION 
### Problem Statement
With the development of deep learning techniques, the size of advanced models is getting larger and larger. For example, GPT-3 model has 175B parameters, and is trained on 500B dataset. These models, while achieving SOTA results, are only available for head players. There are many challenges to apply these models for most users:

1. Time cost and money cost of training an advanced are extremely high when training from scratch. For example, it costs 4.6M $ and takes 355 GPU-years to train GPT-3 model from scratch.
    
2. Training these advanced models requires a large amount of labeled data. For example, the widely used ImageNet-1k dataset has 1.28M labeled images.
   
3. Hardware with limited resources can't enjoy the benefits of advanced models, because advanced models can't deploy on them, e.g., mobile devices.

### Solution with Intel® AIOK Model Adaptor

An intuitive idea is: Can we use publicly available resources to reduce the cost of model training and data collection?  

Over time, more and more publicly available pre-trained models and labeled datasets are emerging on Internet. Then we can transfer “knowledge” from these available pre-trained models and labeled datasets to our target task with several SOTA technologies, i.e., pretraining & finetuning [1], distillation [2], and domain adaption [3].

- Pretraining & Finetuning: During pretraining, an advanced model is trained on a large dataset; During finetuning, the pretrained advanced model is applied to downstream tasks. 

- Distillation: Distillation is used to distill knowledge from pre-trained advanced model (called teacher model) to a different model (called student model). Generally, the output of teacher model serves as pseudo-label for student model, and the student model learns to fit the pseudo-label to perform knowledge transferring. 

- Domain adaption: Domain adaption is used to reuse another dataset (called "source domain dataset") to target task. 

In fact, these technologies can be regarded as regularization strategies:
- Pretraining & Finetuning: The weight of the target model is as close as possible to that of the pretrained model.
- Distillation: The output of the student model is as close as possible to that of the pretrained teacher model.
- Domain adaption:The representation of source domain data is as similar as possible to that of target domain data.

To combine all these technologies and maximize the capability of transfer learning, we provide a Model Adaptor toolkit. 

### Personas the solution is applicable for

1. Individual user can use Model Adaptor to take advantage of advanced models efficiently and effectively. 
   
2. Data scientists can use Model Adaptor to perform knowledge transferring, focusing on where to transfer instead of how to transfer , and improve data analysis.

3. Developers can integrate Model Adaptor to their products, equipping the capability of transfer knowledge.

4. Enterprises can combine Model Adaptor into existing pipeline to improve efficiency and effectiveness.

5. Users without GPU/TPU use can Model Adaptor to efficiently perform knowledge transferring on CPU.

## ARCHITECTURE 
In Model Adaptor, we have implemented all the above three transfer learning technologies with a unified API. Besides, Model Adaptor can be easily integrated with existing pipeline, requiring only a few code changes. Finally, Model Adaptor makes additional efforts on optimization of CPU-training and CPU-inference, both on single-node and multi-node.

Training pipeline with Model Adaptor is shown as following. The colored boxes are Model Adaptor-specific, which is the difference between pipeline with Model Adaptor and the general pipeline. 

**Model Adaptor-specific components**:
- **Pre-trained model**: A pre-trained model is served as a knowledge base to Model Adaptor. Model Adaptor transfer knowledge from pre-trained models to target task.
  Model Adaptor provides many built-in pre-trained models by default, however, users can specify their own pre-trained model.
- **Source Domain data**: Source domain data, which is much bigger than target data,  is served as another knowledge base to Model Adaptor. Model Adaptor transfer knowledge from source domain data to target task. 
  
- **Unified API**: Unified API is a unified interface to assign different transfer learning ability to the underlying model. The Unified API returns a transferable model, which is a wrapper of the underlying model. The type of transferable model is TransferableModel, which shares the same interface with the underlying model.
- **Finetuner**: A finetuner is one of the components for TransferableModel, performs finetuning and servs as a regularizer of underlying model. 
- **Distiller**: A distiller is second of the components for TransferableModel, performs distillation and servs as another regularizer of underlying model. 
- **Adapter**: : An adapter is the third of the components for TransferableModel, performs domain adaption and serves as third regularizer of underlying model.

<img src="./doc/imgs/arch.png" width="800px" />

## Getting Started 
### Environment Setup
   ```bash
   git clone https://github.com/intel/e2eAIOK.git
   cd e2eAIOK
   git submodule update --init --recursive
   python scripts/start_e2eaiok_docker.py -b pytorch112 -w ${host0} ${host1} ${host2} ${host3} --proxy ""
   ```

### Enter Docker
   ```
   sshpass -p docker ssh ${host0} -p 12347
   ```
 
### Pipeline with Model Adaptor

   The general pipeline trains a resnet50 model on CIFAR100 dataset from scratch. See [here](./example/Basic_ResNet50_on_CIFAR100_train.ipynb).  The general pipeline can be enpowered transfer learning ability by Model Adaptor.

#### Finetuning with Model Adaptor

Two steps for finetuning with Model Adaptor: 

1. Download the pretrained resnet50 from [ImageNet-21K Pretraining for the Masses](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth)[4], which is pretrained on Imagenet21k.

2. Create a transferable model with Model Adaptor Unified API:
    ```python
    model = timm.create_model('resnet50', pretrained=False, num_classes=100).to(device)

    pretrained_path = './model-zoo/resnet50_miil_21k.pth' # download path
    pretrained_model = timm.create_model('resnet50', pretrained=False, num_classes=11221).to(device)
    pretrained_model.load_state_dict(torch.load(pretrained_path,map_location=device)["state_dict"], strict=True)
    finetunner= BasicFinetunner(pretrained_model, is_frozen=False)
    model = make_transferrable_with_finetune(model, loss_fn, finetunner)
    ```
    Now, the returned model is transferable, and replaces the original model in general pipeline. 

The full code see [here](./example/pipeline_with_finetuner.py). 

**

#### Distillation with Model Adaptor

Two steps for distillation with Model Adaptor: 

1. Prepare a teacher model, here we select pretrained vit_base-224-in21k-ft-cifar100 is from [HuggingFace](https://huggingface.co/edumunozsala/vit_base-224-in21k-ft-cifar100).

2. Create a transferable model with Model Adaptor Unified API:
   ```python
   model = timm.create_model('resnet50', pretrained=False, num_classes=100).to(device)

   teacher_model = AutoModelForImageClassification.from_pretrained("edumunozsala/vit_base-224-in21k-ft-cifar100")
   distiller= BasicDistiller(teacher_model, num_classes=100)
   model = make_transferrable_with_knowledge_distillation(model, loss_fn, distiller)
   ```
   Now, the returned model is transferable, and replaces the original model in general pipeline. 

The full code see [here](./example/Distiller_ResNet18_from_VIT_on_CIFAR100_train.ipynb). 

** Acceleration with logits saving **
During distillation, teacher forwarding usually takes a lot of time. To accelerate the training procedure, We can save the predicting logits from teacher in advance and reuse it in later student training. Here is the [Logits saving demo](./example/Distiller_ResNet18_from_VIT_on_CIFAR100_save_logits.ipynb) and the code for [training with saved logits](./example/Distiller_ResNet18_from_VIT_on_CIFAR100_train_with_logits.ipynb)

#### Domain Adaption with Model Adaptor

We show how to transfer knowledge from MNIST dataset[5] to USPS dataset [6] with Model Adaptor. MNIST dataset is a large collection of handwritten digits, while USPS dataset is a digit dataset automatically scanned from envelopes by the U.S. Postal Service.

Two steps for domain adaption with Model Adaptor: 

1. Create both source domain dataset and target domain dataset:
    ```python
    data_folder='./dataset' # dataset location
    trainset_source_domain = datasets.MNIST(root=data_folder, train=True, download=True, transform=train_transform_source_domain)
    trainset_target_domain = datasets.USPS(root=data_folder, train=True, download=True, transform=train_transform_target_domain)
    testset = datasets.USPS(root=data_folder, train=False, download=True, transform=test_transform_target_domain)

    train_loader_source_domain = DataLoader(dataset=trainset_source_domain, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    train_loader_target_domain = DataLoader(dataset=trainset_target_domain, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    validate_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    ```
2. Create a transferable model with Model Adaptor Unified API:
   ```python
   model = timm.create_model('resnet18', pretrained=False, num_classes=10).to(device)

   adapter_input_name = 'view'  # which layer in model as adapter input
   adapter_input_size = 32      # adapter input size
   adapter= DANNAdapter(adapter_input_size, adapter_input_name)
   model = make_transferrable_with_domain_adaption(model, loss_fn,adapter,adapter_input_size,adapter_input_name,train_loader_target_domain, train_loader_source_domain)
   ```
   Now, the returned model is transferable, and replaces the original model in general pipeline.

The full code see [here](./example/pipeline_with_adaption.py).  

### Built-in Pipeline

User can create and train a new built-in pipeline with Model Adaptor by a yaml configuration file. The created built-in pipeline is optimized for CPU-training, CPU-inference,  and distributed data parallel training. 

There are [demo](./example/Model_Adapter_Demo.ipynb) about how to training with built-in pipeline with Model Adaptor. 

## Reference
[1] He, K., Girshick, R., Doll´ar, P.: Rethinking imagenet pre-training. In: ICCV (2019)

[2] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015

[3] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, pages 325–333, 2015

[4] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv:2104.10972, 2021

[5] http://yann.lecun.com/exdb/mnist/

[6] https://arxiv.org/abs/1808.01102