#!/bin/bash
#SBATCH --account=lasti
#SBATCH --error=./output/error_%j.log
#SBATCH --output=./output/out_%j.log
#SBATCH --job-name=eae_re
#SBATCH --partition=classicgpu
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G

set -e          # exit on command errors
set -o pipefail # capture fail exit codes in piped commands
#set -u          # treat unset variables and parameters as an error when performing parameter expansion
#set -o xtrace

# $1: code directory
# $2: configuration file
# $3: result directory name
# $4: directory for saving results



echo "=== start"
date
echo "hostname $HOSTNAME"
echo "ppid     $PPID"
echo "cuda     $CUDA_VISIBLE_DEVICES"

source /home/users/atuo/anaconda3/etc/profile.d/conda.sh
conda activate proto_re
python train.py control/train_config.json
date

