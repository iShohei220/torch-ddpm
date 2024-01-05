import argparse
import os
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import torchvision
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms import v2
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
        default=4
    )
    parser.add_argument('--seed', type=int, help='random seed (default: 0)', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--port", type=int, default=12355)

    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train(
    args, 
    model, 
    ema_model,
    optimizer, 
    scheduler, 
    dataloader, 
    writer, 
    epoch, 
    detect_anomaly=False
):
    global_rank = int(os.environ["RANK"])
    model.train()
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)

            global_step = 1000 * epoch + step
            writer.add_hparams(
                {'dataset': args.dataset},
                {f'loss/train/{global_rank}': loss.div(x[0].numel()).item()},
                run_name='.',
                global_step=global_step
            )

    global_step = 1000 * ( epoch + 1 )
    writer.add_image(
        f'ground truth/{global_rank}', 
        make_grid(x, 8, pad_value=1.0), 
        global_step
    )

    ema_model.eval()
    x_hat = ema_model.module.module.sample(x.size(0))
    writer.add_image(
        f'samples/{global_rank}', 
        make_grid(x_hat, 8, pad_value=1.0), 
        global_step
    )


def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def run(args):
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])

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
        lr = 2e-5
        num_epochs = 1200

    else:
        raise NotImplementedError

    num_workers = args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    num_samples = 1000 * args.batch_size

    generator = torch.Generator()
    generator.manual_seed(args.seed + int(os.environ["RANK"]))

    sampler = RandomSampler(
        training_data, 
        replacement=True,
        num_samples=num_samples, 
        generator=generator
    )

    dataloader = DataLoader(
        training_data, 
        batch_size=args.batch_size,
        sampler=sampler, 
        **kwargs
    )

    # Model
    model = DDP(
        DDPM(resolution, in_channels, channels).to(local_rank),
        device_ids=[local_rank]
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, 1.0/5000, 1.0, 5000)

    # EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

    if args.log_dir is None:
        args.log_dir = "runs/" + args.dataset
    writer = SummaryWriter(args.log_dir)

    for epoch in tqdm(range(num_epochs)):
        train(
            args, 
            model, 
            ema_model,
            optimizer, 
            scheduler, 
            dataloader, 
            writer, 
            epoch
        ) 

    if int(os.environ["RANK"]) == 0:
        writer.close()

    cleanup()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    run(args)
