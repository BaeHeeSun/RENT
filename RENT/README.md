### RENT

Please check `bash` folder. The bash file is constructed as 
```
python3 main.py --dataset ${dset} --noise_type ${noise_type} --noisy_ratio ${noise_ratio} --class_method ${cls} --set_gpu ${gpu} --seed ${seed} --data_dir ${DIR}
```
With regard to the `dataset`, CIFAR10/CIFAR100 is supported without any more downloading. 

`class_method` is related to the transition matrix estimation. Currently, supported trasition matrix estimation methods are:
* [Forward](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf)
* [DualT](https://github.com/a5507203/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning)
* [TV](https://github.com/YivanZhang/lio)
* [VolMinNet](https://github.com/xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points)
* [Cycle](https://openreview.net/forum?id=IvnoGKQuXi)

but not limited to those. TForward means learning with the true transition matrix, and only class-conditional noise (sym and asym) is supported.

To check or change arguments, refer to [`argument.py`](https://github.com/BaeHeeSun/RENT/blob/main/RENT/argument.py).
