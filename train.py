import argparse
import os
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.utils import make_grid

from datasets import CelebAHQ
from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser(
        description="Denoising Diffusion Probabilistic Models"
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        help='number of data loading workers', 
        default=os.cpu_count()
    )
    parser.add_argument('--seed', type=int, help='random seed (default: 0)', default=0)
    parser.add_argument("--num_epochs", type=int, default=int(1e7),
                        help="number of epochs (default: 1e7)")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        '--device_ids', 
        type=int, 
        nargs='+', 
        help='list of CUDA devices (default: list(range(torch.cuda.device_count())))', 
        default=list(range(torch.cuda.device_count()))
    )
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--port", type=int, default=12355)

    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train(
    rank, 
    args, 
    model, 
    optimizer, 
    scheduler, 
    dataloader, 
    writer, 
    epoch, 
    detect_anomaly=False
):
    model.train()
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if rank == 0:
                global_step = 1000 * epoch + step
                writer.add_hparams(
                    {'dataset': args.dataset},
                    {'loss/train': loss.div(x[0].numel()).item()},
                    run_name='.',
                    global_step=global_step
                )

        if rank == 0:
            global_step = 1000 * ( epoch + 1 )
            writer.add_image(
                'ground truth', 
                make_grid(x, 8, pad_value=1.0), 
                global_step
            )

            x_hat = model.module.sample(x.size(0))
            writer.add_image(
                'samples', 
                make_grid(x_hat, 8, pad_value=1.0), 
                global_step
            )


def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def setup(rank, args):
    port = args.port
    os.environ['MASTER_ADDR'] = 'localhost'
    while True:
        os.environ['MASTER_PORT'] = str(port)
        try:
            # initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        except RuntimeError:
            port += 1
        else:
            break

def cleanup():
    dist.destroy_process_group()


def run(rank, args):
    setup(rank, args)

    # Dataset
    if args.dataset == "mnist":
        training_data = MNIST(
            "./data",
            download=True,
            transform=v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        )

        in_channels = 1
        resolution = 28
        channels = [64, 128, 256]
        dropout = 0.0
        batch_size = 64
        lr = 2e-4
    
    elif args.dataset == "cifar10":
        training_data = CIFAR10(
            "./data",
            download=True,
            transform=v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        )

        in_channels = 3
        resolution = 32
        channels = [128, 256, 256, 256]
        dropout = 0.1
        batch_size = 128
        lr = 2e-4
        num_epochs = 800

    elif args.dataset == "celebahq":
        training_data = CelebAHQ(
            "./data/celebahq/data256x256",
            transform=v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        )

        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
        dropout = 0.0
        batch_size = 64
        lr = 2e-5
        num_epochs = 500

    elif args.dataset == "lsun_bedroom":
        training_data = LSUN(
            "./data/lsun",
            classes=["bedroom_train"],
            transform=v2.Compose([
                v2.ToImage(),
                v2.CenterCrop(256),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True)
            ]),
        )

        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
        dropout = 0.0
        batch_size = 64
        lr = 2e-5
        num_epochs = 2400

    elif args.dataset == "lsun_church":
        training_data = LSUN(
            "./data/lsun",
            classes=["church_outdoor_train"],
            transform=v2.Compose([
                v2.ToImage(),
                v2.CenterCrop(256),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True)
            ]),
        )

        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
        dropout = 0.0
        batch_size = 64
        lr = 2e-5
        num_epochs = 1200

    else:
        raise NotImplementedError

    batch_size = batch_size // args.world_size
    num_workers = args.num_workers // args.world_size
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    num_samples = 1000 * batch_size

    generator = torch.Generator()
    generator.manual_seed(args.world_size * args.seed + rank)

    sampler = RandomSampler(
        training_data, 
        replacement=True,
        num_samples=num_samples, 
        generator=generator
    )

    dataloader = DataLoader(
        training_data, 
        batch_size=batch_size,
        sampler=sampler, 
        **kwargs
    )

    # Model
    model = DDP(
        DDPM(resolution, in_channels, channels).to(rank),
        device_ids=[rank]
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, 1.0/5000, 1.0, 5000)

    writer = None
    if rank == 0:
        # Tensorboard
        writer = SummaryWriter()


    for epoch in tqdm(range(args.num_epochs)):
        train(
            rank, 
            args, 
            model, 
            optimizer, 
            scheduler, 
            dataloader, 
            writer, 
            epoch
        ) 

    if rank == 0:
        writer.close()

    cleanup()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    mp.spawn(
        run,
        args=(args,),
        nprocs=args.world_size,
        join=True
    )
