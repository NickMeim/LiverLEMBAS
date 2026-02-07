#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH -J LODO_training
#SBATCH --array=0-59%4

# Configuration
module load miniforge/24.3.0-0
mamba activate nikos

# Run LODO training and validation for this drug index
python3 ./lodo_train_validate.py \
    --DrugsIn "../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv" \
    --TargetsIn "../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv" \
    --TFsOut "../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv" \
    --ChemicalSims "../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv" \
    --PKN "../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv" \
    --PknAnnotation "../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv" \
    --infoPath "../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv" \
    --safe_drugs_file "../preprocessing/preprocessed_data/LeaveOneDrugOut/safe_drugs_to_remove_LODO.csv" \
    --signs_file "../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_signs_majority_vote.csv" \
    --res_dir "../results/LODO/" \
    --outPattern "LODO_hepatocytes" \
    --drug_index ${SLURM_ARRAY_TASK_ID} \
    --timepoint "2 hour" \
    --stimulation_strength 100

echo "LODO validation completed for drug index: ${SLURM_ARRAY_TASK_ID}"
