#!/bin/bash
#SBATCH --job-name=gen_inj_file
#SBATCH --output=gen_inj_file.log
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10gb


source ~/.bashrc

#actual end time I want to use: 1238770818

python lvc_rates_injections.py \
    --gps-start-time 1238166018 \
    --gps-end-time 1238186018 \
    --max-redshift 0.05 \
    --time-interval 0 \
    --time-step 100 \
    --mass-distribution UNIFORM_PAIR \
    --min-mass 1 \
    --max-mass 2.6 \
    --spin-distribution ALIGNED \
    --max-spin 0.05 \
    --waveform name \
    --approximant SpinTaylorT4 \
    --min-frequency 30 \
    --max-frequency 1500 \
    --delta-frequency 1 \
    --h1-reference-spectrum-file H1-AVERAGE_PSD-1241560818-28800.txt \
    --l1-reference-spectrum-file L1-AVERAGE_PSD-1241560818-28800.txt \
    --snr-calculation GENERIC \
    --snr-threshold 6 \
