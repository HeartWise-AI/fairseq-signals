# Fairseq-signals

This is an adaption of the [`Fairseq-signals`](https://github.com/Jwoo5/fairseq-signals) for Heartwise.

It is highly recommended to create a dedicated conda environment before installing the following libraries

# Requirements and Installation
* [PyTorch](https://pytorch.org) version >= 1.5.0
* Python version >= 3.6, and <= 3.9
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
git clone https://github.com/HeartWise-AI/fairseq-signals
cd fairseq-signals
pip install pip==24.0
pip install --editable ./
pip install omegaconf==2.0.5 hydra-core==1.0.4
```

* **To preprocess ECG datasets**: `pip install pandas scipy wfdb`
* **To build cython components**: `python setup.py build_ext --inplace`
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`

It is highly recommended to create a dedicated conda environment before installing the following libraries

# Getting Started
For general commands on ecg preprocessing, model pretraining and other fairseq-signals specific, please check [`Fairseq-signals`](https://github.com/Jwoo5/fairseq-signals).
Here we will focus on explaining how to train your own classification models end-to-end or initializing weights from DeepECG-SSL (both linear probing and finetuning),
how to do inference of the models within fairseq-signals and how to save models in onnx formats so they can be used anywhere else for inference. 

## 1: Install fairseq-signals
Here we assumed you cloned the project and install the different modules as described in the Requirements and Installation section
## 2: Finetune DeepECG-SSL
This will update all the weights (base model + classification head)
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] +task.npy_dataset=true model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
- `[device_num]` should be an integer representing the cuda gpu number (starting at 0)
- `[use_fp16]` should be either `true` for `false`. When finetuning `DeepECG-SSL`, use `true`
- `[manifest_folder]` is a folder containing at least two manifest files `train.tsv` and `valid.tsv`, in the section `Manifest structure` we will give example of this file
- `[deepecg-ssl-path]` is the .pt corresponding to `DeepECG-SSL` base model. The `valid` set is used to earlystop the finetuning.
- `[+task.npy_dataset=true]`: in case your training/validation data are save in a panda array, replace this with `[+task.df_dataset=true]`
- `[num_labels]` is an integer corresponding to the number of target class in the classification. For binary classification, `[num_labels] = 1`
- `[loss_name]` correspond to the loss function. Possible values are `as` (asymetric loss), `bf` (binary focal loss), `mse` (mean square error loss), `bce` (binary cross entropy with logits loss), and `mlsml` (multilabel soft marginal loss)
- `[checkpoint_dir]` is the checkpoint directory subfolder name inside the folder `outputs/`. fairseq-signals automatically classify folder using the date and time when they are created

## 3: Linear evaluation with DeepECG-SSL
This will freez the base model weights and only update classification weights
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] model.linear_evaluation=true +task.npy_dataset=[npy_dataset] \
    model.num_labels=[num_labels] criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
All parameters are the same as for finetuning, expect `model.linear_evaluation` whose value is `true`

## 4: End to end model
In case you want to initialized all the weights (base transformer + classification head) randomnly, in other to perform end-to-end training, you the following
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.no_pretrained_weights=true +task.npy_dataset=[npy_dataset] model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
All parameters are the same as for finetuning, expect `model.no_pretrained_weights=true` and `[deepecg-ssl-path]`, the base model path, which is not set, as it is not needed.

## 5: Inference of trained models
Once you trained your model, either by finetuiing, linear evalution or end-to-end, the final checkpoint is saved in the corresponding directory (specified with `[checkpoint_dir]`
Now you can run the inference on a given `.tsv` file that specifies your test data.
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-inference task.data=[manifest_folder] \
    common_eval.path=[model_to_evaluate] common_eval.results_path=[results_path] \
    task.npy_dataset=true model.num_labels=[num_labels] dataset.valid_subset=[test_file] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```
- `[model_to_evaluate]` correspond to the `.pt` of the model we trained and we want to evaluate now
- `[results_path]`, usually set to the same value as `[model_to_evaluate]` correspond to the path were the inference logits will be saved
- `[test_file]` correspond to the `.tsv` we want to evaluate. Usually it is simple `test.tsv`

## 6: Saving trained models on `onnx` format
To be able to run the inference on the trained models outside the fairseq signal environoment, we can use the followwing command
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-save common.fp16=[use_fp16] \
    common_eval.path=[model_to_evaluate] model.num_labels=[num_labels] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```
A file named `model.onnx` will be saved in the same folder as `[model_to_evaluate]`
Once we have this `model.onnx` we can run it everywhere using the following code

```
import onnxruntime as ort
import numpy as np


model_name = 'model.onnx'

def get_session(filename, use_gpu=False):
    if use_gpu:
        return ort.InferenceSession(filename, providers=["CUDAExecutionProvider"])
    else:
        return ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
    

def run_session(session, X):
    """
    session: orn session, obtained from get_session
    X: numpy array of shape (batch_size, 12, 2500) and dtype np.float16 ideally
    """
    input = {
        session.get_inputs()[0].name: X,
    }
    output = session.run(None, input)
    return output


x = 0.00488*np.transpose(np.squeeze(X[0:15]), (0, 2, 1)).astype(np.float16)
use_gpu=  False # can also be set to True
session = get_session(model_name, use_gpu)
y = run_session(session, x)
print(y)
```

