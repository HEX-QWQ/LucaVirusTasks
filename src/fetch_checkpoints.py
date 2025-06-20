import csv
import os
from utils import (
    download_trained_checkpoint_lucaone,
    download_trained_checkpoint_lucavirus,
    download_trained_checkpoint_downstream_tasks,
)

# Static mapping from dataset_name to dataset_type
DATASET_TYPE_MAPPING = {
    # dataset_name: dataset_type
    "RdRP": "protein",
    "ViralCapsid": "protein", 
    "DeepAbBindv2_genome": "protein_protein_gene",
    "DeepAbBindv2_nucl": "protein_protein_gene",
    "DeepAbBindv2_original": "protein_protein_protein",
    "VirusEC4": "protein",
    "DMS_Bind_Reps_Strain": "protein",
    "DMS_Bind_Reps_Strain_Nucl": "gene",
}

# Function to read CSV files and extract checkpoint information
def read_model_manifest(csv_file_path):
    """Read model manifest CSV file and return list of checkpoint dictionaries"""
    checkpoints = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset_name = row['dataset_name']
            
            # Get dataset_type from the static mapping
            dataset_type = DATASET_TYPE_MAPPING[dataset_name]
            
            # Extract required fields for download
            checkpoint = {
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'task_type': row['task_type'],
                'model_type': row['model_type'],
                'input_type': row['input_type'],
                'time_str': row['time_str'],
                'step': int(row['step']) if row['step'] else 0
            }
            
            checkpoints.append(checkpoint)
    
    return checkpoints

def main():
    # Download LucaOne and LucaVirus pretrained models
    download_trained_checkpoint_lucaone(
        llm_dir="../",
        llm_type="lucaone",
        llm_version="v2.0",
        llm_task_level="token_level,span_level,seq_level,structure_level",
        llm_time_str="20231125113045",
        llm_step="17600000",
        base_url="http://47.93.21.181/lucaone/TrainedCheckPoint"
    )

    download_trained_checkpoint_lucavirus(
        llm_dir="../",
        llm_type="lucavirus",
        llm_version="v1.0",
        llm_task_level="token_level,span_level,seq_level",
        llm_time_str="20240815023346",
        llm_step="3800000",
        base_url="http://47.93.21.181/lucavirus/TrainedCheckPoint/"
    )

    # Download downstream task checkpoints
    # Read all model manifests
    model_manifest_dir = "../model_manifest"
    all_checkpoints = []

    # Read binary_class.csv
    binary_class_path = os.path.join(model_manifest_dir, "binary_class.csv")
    if os.path.exists(binary_class_path):
        print(f"Reading {binary_class_path}...")
        binary_checkpoints = read_model_manifest(binary_class_path)
        all_checkpoints.extend(binary_checkpoints)
        print(f"Found {len(binary_checkpoints)} binary classification checkpoints")

    # Read multi_label.csv
    multi_label_path = os.path.join(model_manifest_dir, "multi_label.csv")
    if os.path.exists(multi_label_path):
        print(f"Reading {multi_label_path}...")
        multi_label_checkpoints = read_model_manifest(multi_label_path)
        all_checkpoints.extend(multi_label_checkpoints)
        print(f"Found {len(multi_label_checkpoints)} multi-label checkpoints")

    # Read regression.csv
    regression_path = os.path.join(model_manifest_dir, "regression.csv")
    if os.path.exists(regression_path):
        print(f"Reading {regression_path}...")
        regression_checkpoints = read_model_manifest(regression_path)
        all_checkpoints.extend(regression_checkpoints)
        print(f"Found {len(regression_checkpoints)} regression checkpoints")

    print(f"\nTotal checkpoints found: {len(all_checkpoints)}")

    # Prepare data for download function
    if all_checkpoints:
        # Extract lists for the download function
        dataset_names = [cp['dataset_name'] for cp in all_checkpoints]
        dataset_types = [cp['dataset_type'] for cp in all_checkpoints]
        task_types = [cp['task_type'] for cp in all_checkpoints]
        model_types = [cp['model_type'] for cp in all_checkpoints]
        input_types = [cp['input_type'] for cp in all_checkpoints]
        time_strs = [cp['time_str'] for cp in all_checkpoints]
        steps = [cp['step'] for cp in all_checkpoints]
        
        print("\nDownloading all downstream task checkpoints...")
        print(f"Number of checkpoints to download: {len(all_checkpoints)}")
        
        # Download all checkpoints
        download_trained_checkpoint_downstream_tasks(
            save_dir="../",
            dataset_name=dataset_names,
            dataset_type=dataset_types,
            task_type=task_types,
            model_type=model_types,
            input_type=input_types,
            time_str=time_strs,
            step=steps,
            base_url="http://47.93.21.181/lucavirus/DownstreamTasksTrainedModels"
        )
        
        print("\nAll downloads completed!")
    else:
        print("No checkpoints found in model manifests.")

if __name__ == "__main__":
    main()