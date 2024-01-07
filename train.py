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
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument(
        '--save_every', 
        type=int, 
        help='How often to save a snapshot (default: 100)', 
        default=100
    )

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
    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()

        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(model)

        global_step = 10000 * epoch + step
        writer.add_scalar(
            f'loss/train/{global_rank}',
            loss.div(x[0].numel()).item(),
            global_step
        )

    global_step = 10000 * ( epoch + 1 )
    writer.add_image(
        f'ground_truth/{global_rank}', 
        make_grid(x, 8, pad_value=1.0), 
        global_step
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed + int(os.environ["RANK"]))

    ema_model.eval()
    x_hat = ema_model.module.module.sample(x.size(0), generator)

    writer.add_image(
        f'samples/{global_rank}', 
        make_grid(x_hat, 8, pad_value=1.0), 
        global_step
    )

    if global_rank == 0:
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EMA_MODEL_STATE": ema_model.module.state_dict(),
            "OPTIMIZER_STATE": optimizer.state_dict(),
            "SCHEDULER_STATE": scheduler.state_dict(),
            "EPOCHS_RUN": epoch+1,
        }
        torch.save(snapshot, args.snapshot_path)


def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def run(args):
    setup()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Dataset
    if args.dataset == "cifar10":
        transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        if global_rank == 0:
            training_data = CIFAR10(
                "./data",
                download=True,
                transform=transform
            )

        dist.barrier()
        if global_rank != 0 and local_rank == 0:
            training_data = CIFAR10(
                "./data",
                download=True,
                transform=transform
            )

        dist.barrier()
        if local_rank != 0:
            training_data = CIFAR10(
                "./data",
                download=False,
                transform=transform
            )

        in_channels = 3
        resolution = 32
        channels = [128, 256, 256, 256]
        dropout = 0.1
        lr = 2e-4
        num_epochs = 80

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
        num_epochs = 50

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
        num_epochs = 240

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
        num_epochs = 120

    else:
        raise NotImplementedError

    num_workers = args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    num_samples = 10000 * args.batch_size

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
        args.log_dir = os.path.join("runs", f"{args.dataset}-{args.seed}")
    writer = SummaryWriter(args.log_dir)
    args.snapshot_path = os.path.join(args.log_dir, "snapshot.pt")

    epoch_start = 0
    if os.path.exists(args.snapshot_path):
        loc = f"cuda:{local_rank}"
        snapshot = torch.load(args.snapshot_path, map_location=loc)
        model.load_state_dict(snapshot["MODEL_STATE"])
        ema_model.load_state_dict(snapshot["EMA_MODEL_STATE"])
        optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        epoch_start = snapshot["EPOCHS_RUN"]

    for epoch in tqdm(range(epoch_start, num_epochs)):
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

    writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    run(args)
