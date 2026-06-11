#!/bin/bash
export SBATCH_ACCOUNT=lcls:rix101231825

CONFIG=/sdf/data/lcls/ds/rix/rix101231825/results/smalldata_tools/lcls2_producers/prod_config_crix.py
SUBMIT=/sdf/data/lcls/ds/rix/rix101231825/results/smalldata_tools/arp_scripts/submit_smd2.sh
SMD_DIR=/sdf/data/lcls/ds/rix/rix101231825/hdf5/smalldata

# loop through all the runs and process them
"$SUBMIT" --partition milano --nodes 3 --config crix \
                --gather_interval 50 --experiment rix101231825 --s3df \
                --directory "$SMD_DIR" --run "112"

