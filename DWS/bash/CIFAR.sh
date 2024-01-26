cd ..

gpu=0
DIR=/home/aailab/data/baeheesun/noise_generalization/data/
dset=CIFAR10
alpha=10

for seed in 0 1 2 3 4
do
for cls in Forward_Weight DualT_Weight TV_Weight VolMinNet_Weight Cycle_Weight TReweight
do
python3 main.py --dataset CIFAR10 --noise_type sym --noisy_ratio 0.2 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --alpha ${alpha} --data_dir ${DIR}
python3 main.py --dataset CIFAR10 --noise_type sym --noisy_ratio 0.5 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --alpha ${alpha} --data_dir ${DIR}
python3 main.py --dataset CIFAR10 --noise_type asym --noisy_ratio 0.2 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --alpha ${alpha} --data_dir ${DIR}
python3 main.py --dataset CIFAR10 --noise_type asym --noisy_ratio 0.4 --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --alpha ${alpha} --data_dir ${DIR}
done
done