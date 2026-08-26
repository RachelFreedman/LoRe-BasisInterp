# full grid across seeds — the main reproducibility check (Output in replication_sweep_results.csv)
python replication_sweep.py --k-values 1,5,10,25,50 --seed-values 0,1,2,42

# v_sft-init ablation
python replication_sweep.py --k-values 1,10,50 --seed-values 42 --init vsft   --out sweep_vsft.csv
python replication_sweep.py --k-values 1,10,50 --seed-values 42 --init random --out sweep_random.csv

# Alpha sweep
for a in 0 1 5 10 25 50 100 1000; do python replication_sweep.py --k-values 10 --seed-values 42 --alpha $a --out a$a.csv; done

# Log training every 500 steps
python log_training_curves.py --alphas 0,100,1000,10000 --K 10 --seed 42 --log-every 500 --out training_curves.csv
python log_training_curves.py --alphas 0,10,25,50 --K 10 --seed 42 --log-every 500 --out training_curves1.csv

# generate training curves
python plot_training_curves.py training_curves*.csv --out combined.png
