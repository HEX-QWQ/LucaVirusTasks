# LucaVirus Downstream Tasks

This repository contains code for training and inference of downstream prediction tasks leveraging LucaVirus and other biological foundation models (e.g. LucaOne, ESM2, ESMC).

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Model Weights & Datasets](#model-weights--datasets)
- [Usage](#usage)
  - [Embedding Inference](#embedding-inference)
  - [Model Inference](#model-inference)
  - [Model Training](#model-training)
- [Contributors](#contributors)
- [Citation](#citation)

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/LucaOne/LucaVirusTasks.git
cd LucaVirusTasks

# 2. Setup environment
conda create -n lucavirus_tasks python=3.9.13
conda activate lucavirus_tasks
pip install -r requirements.txt

# 3. Fetch all checkpoints for both foundation models and downstream tasks
cd src
python fetch_checkpoints.py

# 3. Run inference example
python predict.py \
    --input_file ../dataset/RdRP/protein/binary_class/test/test.csv \
    --seq_type prot \
    --model_path .. \
    --dataset_name RdRP \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20250329135221 \
    --step 83496 \
    --save_path test_output.csv \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0
```

## 🔧 Environment Setup

### Installation Steps

#### 1. Update Git
```bash
# CentOS
sudo yum update && sudo yum install git-all

# Ubuntu
sudo apt-get update && sudo apt install git-all
```

#### 2. Install Python Environment
```bash
# Download Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Install and setup
sh Anaconda3-2022.05-Linux-x86_64.sh
source ~/.bashrc

# Create environment
conda create -n lucavirus_tasks python=3.9.13
conda activate lucavirus_tasks
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Model Weights & Datasets

### Foundation Models
- **LucaVirus**: For pretrain weights, please refer to [LucaVirus repository](https://github.com/LucaOne/LucaVirus)
- **LucaOne**: For pretrain weights, please refer to [LucaOne repository](https://github.com/LucaOne/LucaOne)
- **ESM2/ESMC**: Download automatically when runing training or inference.

Place foundation model weights and logs in `<project_root>/llm/` folder.

### Downstream Models
- Trained model checkpoints available at: `http://47.93.21.181/lucavirus/DownstreamTasksTrainedModels/` and in Zenodo (coming soon). All available checkpoints will be downloaded automatically when running the src/predict.py script, about 2.6Gb in total. If automatic downloading failed, users may download manually from the above URL.
- Model manifest: `model_manifest/*.csv`
- Performance metrics and configuration details included in manifest files

### Datasets
- Training datasets available at: `http://47.93.21.181/lucavirus/DownstreamTasksDatasets/` and in Zenodo (coming soon).
- Pretraining datasets: Refer to [LucaVirus repository](https://github.com/LucaOne/LucaVirus)

## 💻 Usage

### Embedding Inference

Generate embeddings from biological sequences using foundation models.

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--embedding_type` | Output format: `matrix` or `vector` | `matrix` |
| `--seq_type` | Sequence type: `gene` (DNA/RNA) or `prot` (protein) | - |
| `--truncation_seq_length` | Max sequence length for embedding | `4096` |
| `--embedding_complete` | Enable complete sequence embedding | `False` |
| `--gpu_id` | GPU ID to use (`-1` for CPU) | `0` |

#### Usage Examples

**Protein Sequences (CSV format):**
```bash
cd src/llm/lucavirus
python get_embedding.py \
    --llm_dir ../../../ \
    --llm_type lucavirus \
    --seq_type prot \
    --input_file ../../../data/proteins.csv \
    --id_idx 0 --seq_idx 1 \
    --save_path ../../../embeddings/proteins \
    --embedding_type matrix \
    --embedding_complete \
    --gpu_id 0
```

**DNA/RNA Sequences (FASTA format):**
```bash
python get_embedding.py \
    --llm_dir ../../../ \
    --llm_type lucavirus \
    --seq_type gene \
    --input_file ../../../data/sequences.fasta \
    --save_path ../../../embeddings/sequences \
    --embedding_type matrix \
    --embedding_complete \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 0
```

#### Performance Tips

1. **GPU Memory**: Use A100/H100/H200 for large sequences (up to 4096 tokens)
2. **Long Sequences**: Enable `--embedding_complete` and `--embedding_complete_seg_overlap`
3. **DNA Sequences**: Set `--embedding_fixed_len_a_time` (e.g., 4096 for A100)
4. **Protein Sequences**: Usually don't need fixed length setting
5. **CPU Fallback**: Use `--gpu_id -1` for CPU inference

### Model Inference

Run predictions using trained downstream models.

#### Quick Example
```bash
# RdRP prediction
cd src
python predict.py \
    --input_file ../dataset/RdRP/protein/binary_class/test/test.csv \
    --seq_type prot \
    --model_path .. \
    --dataset_name RdRP \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20250329135221 \
    --step 83496 \
    --save_path test_output.csv \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0
```

#### Supported Tasks
- **RdRP Prediction**: Viral RNA-dependent RNA polymerase identification and classification
- **Viral Capsid Prediction**: Viral capsid protein identification and structure prediction
- **Enzymatic Activity Prediction**: Protein enzymatic activity classification and prediction
- **Virus Evolvability Prediction**: SARS-CoV-2 RBD binding affinity landscape prediction for evolutionary analysis
- **Antibody-Antigen Binding Prediction**: SARS-CoV-2 spike protein antibody binding affinity prediction

### Model Training

Train custom downstream models on your datasets.

#### Training Script Structure
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seed=1221

# for dataset
DATASET_NAME="RdRP"
DATASET_TYPE="protein"

# for task
TASK_TYPE="binary_class"
TASK_LEVEL_TYPE="seq_level"
LABEL_TYPE="RdRP"

# for input
INPUT_TYPE="matrix"
INPUT_MODE="single"
TRUNC_TYPE="right"

# for model
MODEL_TYPE="luca_base"
CONFIG_NAME="luca_base_config.json"
FUSION_TYPE="concat"
dropout_prob=0.1
fc_size=128
classifier_size=$fc_size
BEST_METRIC_TYPE="f1"
loss_type="bce"

# for sequence channel
SEQ_MAX_LENGTH=4096
hidden_size=1024
num_attention_heads=0
num_hidden_layers=0
SEQ_POOLING_TYPE="value_attention"
VOCAB_NAME="gene_prot"

# for embedding channel
embedding_input_size=2560
matrix_max_length=4096
MATRIX_POOLING_TYPE="value_attention"

# for llm
llm_type="lucaone_virus"
llm_task_level="token_level,span_level,seq_level"
llm_version="v1.0"
llm_time_str=20240815023346
llm_step=3800000

# for training
num_train_epochs=10
gradient_accumulation_steps=1
logging_steps=200
save_steps=-1
evaluate_strategy=epoch
evaluate_steps=$save_steps

warmup_steps=200
max_steps=-1
batch_size=16
learning_rate=1e-4
buffer_size=1024
pos_weight=40

time_str=$(date "+%Y%m%d%H%M%S")
cd ../../
python run.py \
  --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/train/ \
  --dev_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/dev/ \
  --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/test/ \
  --dataset_name $DATASET_NAME \
  --dataset_type $DATASET_TYPE \
  --task_type $TASK_TYPE \
  --task_level_type $TASK_LEVEL_TYPE \
  --model_type $MODEL_TYPE \
  --input_type $INPUT_TYPE \
  --input_mode $INPUT_MODE \
  --label_type $LABEL_TYPE \
  --alphabet $VOCAB_NAME \
  --label_filepath ../dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/label.txt  \
  --output_dir ../models/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --log_dir ../logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --tb_log_dir ../tb-logs/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$INPUT_TYPE/$time_str \
  --config_path ../config/$MODEL_TYPE/$CONFIG_NAME \
  --seq_vocab_path $VOCAB_NAME \
  --seq_pooling_type $SEQ_POOLING_TYPE \
  --matrix_pooling_type $MATRIX_POOLING_TYPE \
  --fusion_type $FUSION_TYPE \
  --do_train \
  --do_eval \
  --do_predict \
  --do_metrics \
  --evaluate_during_training \
  --per_gpu_train_batch_size=$batch_size \
  --per_gpu_eval_batch_size=$batch_size \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=$learning_rate \
  --lr_update_strategy step \
  --lr_decay_rate 0.9 \
  --num_train_epochs=$num_train_epochs \
  --overwrite_output_dir \
  --seed $seed \
  --sigmoid \
  --loss_type $loss_type \
  --loss_reduction meanmean \
  --best_metric_type $BEST_METRIC_TYPE \
  --seq_max_length=$SEQ_MAX_LENGTH \
  --embedding_input_size $embedding_input_size \
  --matrix_max_length=$matrix_max_length \
  --trunc_type=$TRUNC_TYPE \
  --no_token_embeddings \
  --no_token_type_embeddings \
  --no_position_embeddings \
  --pos_weight $pos_weight \
  --buffer_size $buffer_size \
  --llm_dir .. \
  --llm_type $llm_type \
  --llm_version $llm_version \
  --llm_task_level $llm_task_level \
  --llm_time_str $llm_time_str \
  --llm_step $llm_step \
  --ignore_index -100 \
  --hidden_size $hidden_size \
  --num_attention_heads $num_attention_heads \
  --num_hidden_layers $num_hidden_layers \
  --dropout_prob $dropout_prob \
  --classifier_size $classifier_size \
  --vector_dirpath ../vectors/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$llm_version/$llm_type/$llm_time_str/$llm_step   \
  --matrix_dirpath ../matrices/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/$MODEL_TYPE/$llm_version/$llm_type/$llm_time_str/$llm_step  \
  --seq_fc_size null \
  --matrix_fc_size $fc_size \
  --vector_fc_size null \
  --emb_activate_func gelu \
  --fc_activate_func gelu \
  --classifier_activate_func gelu \
  --warmup_steps $warmup_steps \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight_decay 0.01 \
  --save_steps $save_steps \
  --max_steps $max_steps \
  --logging_steps $logging_steps \
  --evaluate_steps $evaluate_steps \
  --evaluate_strategy $evaluate_strategy \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 3072 \
  --matrix_add_special_token \
  --save_all
```

the above script expect an training data set in <project_root>/dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE folder, and there should be train, test and dev folder within that folder. training data in csv format are placed in train, test and dev folders, and these csv dataset can be splited in to multiple csv file (recommanded when dataset is very large) and the script can handle both single and multiple csv files automatically. for dataset format guideline, please refer to example toy dataset in dataset folder.

#### Key Hyperparameters Description

The training script contains several categories of hyperparameters that control different aspects of the model and training process:

##### **Dataset and Task Configuration**
- `DATASET_NAME`: Name of the dataset (e.g., "RdRP", "DeepVirID", "PPC")
- `DATASET_TYPE`: Type of biological data ("protein", "gene", "nucleotide")
- `TASK_TYPE`: Classification task type ("binary_class", "multi_class", "multi_label")
- `TASK_LEVEL_TYPE`: Prediction level ("seq_level", "token_level", "span_level")
- `LABEL_TYPE`: Specific label category for the task

##### **Input Processing Parameters**
- `INPUT_TYPE`: Input data format ("matrix", "sequence", "vector")
- `INPUT_MODE`: Processing mode ("single", "pair", "triple")
- `TRUNC_TYPE`: Sequence truncation strategy ("right", "left", "middle")
- `SEQ_MAX_LENGTH`: Maximum sequence length (default: 4096)
- `matrix_max_length`: Maximum matrix length for embedding inputs (default: 4096)
- `embedding_input_size`: Dimension of pre-computed embeddings (default: 2560)

##### **Model Architecture Parameters**
- `MODEL_TYPE`: Model variant ("luca_base", "luca_pair", "luca_triple", etc.)
- `hidden_size`: Hidden dimension size (default: 1024)
- `num_attention_heads`: Number of attention heads (0 for no attention layers)
- `num_hidden_layers`: Number of transformer layers (0 for no transformer layers)
- `classifier_size`: Size of the final classification layer (default: 128)
- `dropout_prob`: Dropout rate for regularization (default: 0.1)

##### **Pooling and Fusion Parameters**
- `SEQ_POOLING_TYPE`: Sequence pooling method ("value_attention", "mean", "max", "cls")
- `MATRIX_POOLING_TYPE`: Matrix pooling method for embeddings
- `FUSION_TYPE`: How to combine different input channels ("concat", "sum", "attention")

##### **Training Optimization Parameters**
- `learning_rate`: Initial learning rate (default: 1e-4)
- `batch_size`: Training batch size per GPU (default: 16)
- `num_train_epochs`: Total training epochs (default: 10)
- `warmup_steps`: Learning rate warmup steps (default: 200)
- `gradient_accumulation_steps`: Gradient accumulation for larger effective batch size
- `weight_decay`: L2 regularization coefficient (default: 0.01)
- `beta1`, `beta2`: Adam optimizer parameters (default: 0.9, 0.98)

##### **Loss Function Parameters**
- `loss_type`: Loss function type ("bce", "ce", "focal_loss", "asl")
- `pos_weight`: Positive class weight for imbalanced datasets (default: 40)
- `BEST_METRIC_TYPE`: Evaluation metric for model selection ("f1", "accuracy", "auc")

##### **LLM Integration Parameters**
- `llm_type`: Pre-trained language model type ("lucaone_virus", "esm2", "dnabert2")
- `llm_version`: Version of the pre-trained model
- `llm_task_level`: Task levels supported by the LLM
- `llm_time_str`, `llm_step`: Specific checkpoint identifiers

##### **Activation Functions**
- `emb_activate_func`: Activation for embedding layers ("gelu", "relu", "tanh")
- `fc_activate_func`: Activation for fully connected layers
- `classifier_activate_func`: Activation for classifier layer

##### **Training Control Parameters**
- `seed`: Random seed for reproducibility
- `logging_steps`: Logging frequency during training
- `save_steps`: Model checkpoint saving frequency (-1 for epoch-based)
- `evaluate_strategy`: Evaluation strategy ("epoch", "steps")
- `buffer_size`: Data loading buffer size for performance optimization

These hyperparameters can be tuned based on your specific dataset characteristics, computational resources, and performance requirements. The default values are generally suitable for most biological sequence classification tasks, but may need adjustment for optimal performance on specific datasets.


## 👥 Contributors

- [Yong He](https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en)
- [Yuan-Fei Pan](https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ)
- [Zhaorong Li](https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en)
- [Mang Shi](https://scholar.google.com/citations?user=1KJOH7YAAAAJ&hl=zh-CN&oi=ao)
- Yuqi Liu

## 📚 Citation

If you use this code in your research, please cite this repository. A preprint will be available soon on bioRxiv, please see update in this repository.

**References:**
- **LucaOne**: He, Y., Fang, P., Shan, Y. et al. Generalized biological foundation model with unified nucleic acid and protein language. *Nat Mach Intell* (2025). https://doi.org/10.1038/s42256-025-01044-4
- **ESM2**: Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.
- **ESMC**: ESM Team. (2024). ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning. [EvolutionaryScale Blog](https://evolutionaryscale.ai/blog/esm-cambrian).

---

**Note**: Model weights and datasets will be made publicly available upon publication. Please check this repository and the [LucaVirus repository](https://github.com/LucaOne/LucaVirus) for updates.
