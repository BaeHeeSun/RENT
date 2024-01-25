import os
import torch
import numpy as np

from data import dataloader
import library
import utils

if __name__ == '__main__':
    from argument import args

    print('='*50)
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.set_gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define loader
    args.Loader = dataloader.dataloader(args)

    # Define device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define network
    args.network = utils.define_network(args).to(args.device)
    # Define result save directory
    leaf_dir = os.path.join(args.dataset, args.noise_type+'_'+args.noisy_ratio,args.class_method,'epoch_{eph}_seed_{seed}'.format(eph=args.total_epochs, seed=args.seed))
    args.result_dir = os.path.join('result',leaf_dir)
    args.model_dir = os.path.join('trained_model', leaf_dir)

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    model = library.__dict__[args.class_method](args)
    model.run()

    # Save config
    f = open(os.path.join(args.result_dir,'configs.txt'),'w')
    for arg in vars(args):
        if 'network' in arg:
            continue
        f.write("{} : {}".format(arg, getattr(args, arg)))
        f.write('\n')
    f.close()
