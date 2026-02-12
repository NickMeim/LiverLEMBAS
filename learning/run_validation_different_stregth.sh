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
#SBATCH -J LODO_valdiating
#SBATCH --array=0-59%4

# Configuration
module load miniforge/24.3.0-0
mamba activate nikos

# Run LODO training and validation for this drug index
python3 ./lodo_batch_validate_strengths.py \
    --safe_drugs_file "../preprocessing/preprocessed_data/LeaveOneDrugOut/safe_drugs_to_remove_LODO.csv" \
    --signs_file "../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_signs_majority_vote.csv" \
    --model_dir "../results/LODO/2 hour/" \
    --drug_index ${SLURM_ARRAY_TASK_ID} \
    --timepoint "2 hour" \
    --strengths 50 100 250 500 1000 1500 2000 3000

echo "LODO validation completed for drug index: ${SLURM_ARRAY_TASK_ID}"
