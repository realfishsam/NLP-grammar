#!/bin/sh
#SBATCH --job-name=CS4248_Finetune
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fikri@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gpus=1


srun finetune
fikri@xlogin0:~/jobdata$