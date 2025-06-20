# Task1.1 Revealing viral dark matter: RdRP prediction
LLM_TYPE=lucaone_virus
TIME_STR=20250329135221
STEP=83496

python predict.py \
    --input_file "$1" \
    --seq_type prot \
    --model_path .. \
    --dataset_name RdRP \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task1.2 Revealing viral dark matter: Viral Capsid prediction
LLM_TYPE=lucaone_virus
TIME_STR=20250103142554
STEP=81280

python predict.py \
    --input_file "$1" \
    --seq_type prot \
    --model_path .. \
    --dataset_name ViralCapsid \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task2 Characterizing unknown viral proteins: enzymatic activity prediction (EC number)
LLM_TYPE=lucaone_virus
TIME_STR=20250501135254
STEP=208908

python predict.py \
    --input_file "$1" \
    --seq_type prot \
    --model_path .. \
    --dataset_name VirusEC4 \
    --dataset_type protein \
    --task_type multi_label \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task3.1 Predicting virus evolvability: SARS-CoV-2 binding affinity prediction (protein input)
LLM_TYPE=lucaone_virus
TIME_STR=20250225170110
STEP=104321

python predict.py \
    --input_file "$1" \
    --seq_type prot \
    --model_path .. \
    --dataset_name DMS_Bind_Reps_Strain \
    --dataset_type protein \
    --task_type regression \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task3.2 Predicting virus evolvability: SARS-CoV-2 binding affinity prediction (nucleotide input)
LLM_TYPE=lucaone_virus
TIME_STR=20241225121917
STEP=72386

python predict.py \
    --input_file "$1" \
    --seq_type gene \
    --model_path .. \
    --dataset_name DMS_Bind_Reps_Strain_Nucl \
    --dataset_type gene \
    --task_type regression \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task4.1 Predicting Ab-Ag binding: SARS-CoV-2 Spike antibody binding prediction (spike protein input)
LLM_TYPE=lucaone_virus
TIME_STR=20250117131337
STEP=86059

python predict.py \
    --input_file "$1" \
    --seq_type_a prot \
    --seq_type_b prot \
    --seq_type_c prot \
    --model_path .. \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple2 \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task4.2 Predicting Ab-Ag binding: SARS-CoV-2 Spike antibody binding prediction (spike nucleotide input)
LLM_TYPE=lucaone_virus
TIME_STR=20250127174006
STEP=71366

python predict.py \
    --input_file "$1" \
    --seq_type_a prot \
    --seq_type_b prot \
    --seq_type_c gene \
    --model_path .. \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple2 \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0

# Task4.3 Predicting Ab-Ag binding: SARS-CoV-2 Spike antibody binding prediction (whole viral genome nucleotide input)
LLM_TYPE=lucaone_virus
TIME_STR=20250124160316
STEP=29386

python predict.py \
    --input_file "$1" \
    --seq_type_a prot \
    --seq_type_b prot \
    --seq_type_c gene \
    --model_path .. \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple2 \
    --input_type matrix \
    --input_mode single \
    --time_str $TIME_STR \
    --step $STEP \
    --save_path "$2" \
    --llm_truncation_seq_length 4096 \
    --print_per_num 1000 \
    --gpu_id 0