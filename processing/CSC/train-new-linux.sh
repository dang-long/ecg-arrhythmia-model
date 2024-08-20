#!/bin/bash
#SBATCH --account=project_2010942
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1000
#SBATCH --gres=gpu:v100:2,nvme:10

srun python3.9 "/projappl/project_2010942/ecg-arrhythmia-model/main.py"  --data-dir "/scratch/project_2010942/data/op_08_classes/train_dataset" --epochs 50 --num-workers 2 --batch-size 8 --num-classes 8 --use-gpu
srun python3.9 "/projappl/project_2010942/ecg-arrhythmia-model/main.py"  --data-dir "/scratch/project_2010942/data/op_09_classes/train_dataset" --epochs 50 --num-workers 2 --batch-size 8 --num-classes 9 --use-gpu