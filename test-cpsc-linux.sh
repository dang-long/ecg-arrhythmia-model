#!/bin/bash
#SBATCH --account=project_2010942
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=500
#SBATCH --gres=gpu:v100:1,nvme:10

srun python3.9 "/projappl/project_2010942/ecg-arrhythmia-model/main.py"  --data-dir "/scratch/project_2010942/data/cpsc_processed" --epochs 50 --num-workers 2 --batch-size 8 --num-classes 8 --phase 'test' --test-dir "/scratch/project_2010942/data/op_08_classes/test_dataset"
srun python3.9 "/projappl/project_2010942/ecg-arrhythmia-model/main.py"  --data-dir "/scratch/project_2010942/data/cpsc_processed" --epochs 50 --num-workers 2 --batch-size 8 --num-classes 9 --phase 'test' --test-dir "/scratch/project_2010942/data/op_09_classes/test_dataset"
