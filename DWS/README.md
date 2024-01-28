### DWS

Similar to **RENT** implementation, please check `bash` folder. 
```
python3 main.py --dataset ${dset} --noise_type ${noise_type} --noisy_ratio ${noise_ratio} --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
```
With regard to the `dataset`, CIFAR10/CIFAR100 is supported without any more downloading. 

To check or change arguments, refer to [`argument.py`](https://github.com/BaeHeeSun/RENT/blob/main/DWS/argument.py).
