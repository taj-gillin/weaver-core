#!/bin/bash
#SBATCH --job-name=check_branches
#SBATCH --output=logs/check_branches_%j.out
#SBATCH --error=logs/check_branches_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=batch
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Check for class label branches
python -c "
import uproot
f = uproot.open('/HEP/data/share/aleph/aleph-data/ntuples_new/mc/output_qqb_0_test.root')
tree = f['tree']
print('Looking for class label branches:')
for branch in sorted(tree.keys()):
    if 'is' in branch.lower() or 'label' in branch.lower() or 'class' in branch.lower():
        print(f'  {branch}')
"
