cd ..
gpu=0
DIR=path/to/dataset
dset=CIFAR10

for seed in 5 6 7 8 9
do
for cls in Forward_Sample DualT_Sample TV_Sample VolMinNet_Sample Cycle_Sample TForward_Sample
do
python3 main.py --dataset ${dset} --noise_type sym --noisy_ratio 0.2 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
python3 main.py --dataset ${dset} --noise_type sym --noisy_ratio 0.5 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
python3 main.py --dataset ${dset} --noise_type asym --noisy_ratio 0.2 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
python3 main.py --dataset ${dset} --noise_type asym --noisy_ratio 0.4 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
done
done
