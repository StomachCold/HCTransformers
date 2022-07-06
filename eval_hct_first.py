import os
import argparse
import h5py
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from testCos import testCos

import utils
import vision_transformer_attn as vits

server_dict = {
    'mini':{
        'dataset': 'mini',
        'data_path': '/path/to/mini_imagenet/',     # Need to modify here
        'ckp_path': '/path/to/checkpoint_mini/'},   # Need to be passed in commandline args
    'fs':{
        'dataset': 'fs',
        'data_path': '/path/to/CIFAR-FS/',
        'ckp_path': '/path/to/checkpoint_fs/'},
    'fc100':{
        'dataset': 'fc100',
        'data_path': '/path/to/FC100/',
        'ckp_path': '/path/to/checkpoint_fc100/'},
    'tiered':{
        'dataset': 'tiered',
        'data_path': '/path/to/tiered_imagenet/',
        'ckp_path': '/path/to/checkpoint_tiered/'},
}


def eval_linear(args):
    server = server_dict[args.server]
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_test = datasets.ImageFolder(os.path.join(server['data_path'], args.partition), transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_test)} test imgs.")

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    # args.pretrained_weights = args.pretrained_weights[:-4] + str(args.epochs) + args.pretrained_weights[-4:]

    print(f"Model {args.arch} built.")
    print(f"ckp_path: {server['ckp_path']}")
    
    args.output_dir = server['ckp_path']

    checkdir = os.listdir(server['ckp_path'])
    checkdir.sort()
    checkdir = [checkdir[i] for i in range(len(checkdir)) if '.pth' in checkdir[i]]
    for i in range(len(checkdir)):
        if str(args.epochs) in checkdir[i]:
            if args.epochs != -1:
                checkdir =  checkdir[i:] + checkdir[0:1]
            else:
                checkdir = checkdir[0:1] + checkdir[i:]
            break

    print(f"checkpoints: {checkdir}")

    # pretrained_weights = server['pretrained_weights']
    # checkpoint_key = ['teacher','student']

    for i in range(len(checkdir)):
        print(f"Evaluating pretrained weight in {checkdir[i]}")
        if '.pth' in checkdir[i]:
            args.pretrained_weights = os.path.join(server['ckp_path'],checkdir[i])
            # server['pretrained_weights'] = pretrained_weights + checkdir[i]
            
            if not checkdir[i][-8:-4].isdigit():
                epoch = int(torch.load(args.pretrained_weights)['epoch']) - 1
            else:
                epoch = int(checkdir[i][-8:-4])

            outfile = os.path.join(server['ckp_path'],'{}_224_{}_{}.hdf5'.format(args.partition,epoch, args.checkpoint_key))
            if not os.path.isfile(outfile) or args.isfile == 1:
                utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch,
                                              args.patch_size)
                if args.save == 1:
                    save_features(model,server['dataset'], test_loader, 1, args.avgpool_patchtokens, epoch, server['ckp_path'],outfile)

            testCos(args,server,epoch,server['ckp_path'],outfile)
        if int(args.epochs) == -1:
            return


def save_features(model,dataset,loader, n, avgpool,epochs, pretrained_weights,outfile):
    # outfile = pretrained_weights+'{}_224_{}_{}.hdf5'.format(args.partition,epochs, args.checkpoint_key)
    print('outputfile:',outfile)
    # if os.path.isfile(outfile):
    #     return
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    f = h5py.File(outfile, 'w')
    max_count = len(loader) * loader.batch_size
    print(max_count)
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    # all_attns = None
    count = 0
    for i, (inp, target) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = [x[:,0] for x in intermediate_output]
                if avgpool:
                    output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
                output = torch.cat(output, dim=-1)
            else:
                output = model(inp)

        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(loader)))
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(output.size()[1:]), dtype='f')
        # if all_attns is None:
            # all_attns = f.create_dataset('all_attns', [max_count] + list(attn_output.size()[1:]), dtype='f')
        all_feats[count:count + output.size(0)] = output.data.cpu().numpy()
        all_labels[count:count + output.size(0)] = target.cpu().numpy()
        # all_attns[count:count + attn_output.size(0)] = attn_output.data.cpu().numpy()
        count = count + output.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()
    print(outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation HCTransformer\'s first stage with cosine classification on Few-shot Datasets')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')

    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=300, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    # parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    
    # few-shot args
    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--num_shots', default=1, type=int)
    parser.add_argument('--seed', default=777, type=int)

    # evaluation args
    parser.add_argument('--partition', default='val', type=str)
    parser.add_argument('--epochs', default='-1', type=str, help='Number of epochs of training.')
    parser.add_argument('--save', default=1, type=int)
    parser.add_argument('--isfile', default=-1, type=int)
    parser.add_argument('--server', default='mini', type=str,
                        help='mini / tiered / fs / fc100')
    parser.add_argument('--n',default=1)
    parser.add_argument('--both',default=1, type=int)
    parser.add_argument('--ckp_path',default='',type=str,
                        help='path to the checkpoint of hct')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    
    args = parser.parse_args()
    # setup ckp_path
    server_dict[args.server]['ckp_path'] = args.ckp_path
    
    eval_linear(args)
