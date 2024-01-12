from glob import glob
import math
from multiprocessing import Pool

from cleanfid import fid
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision

from ddpm import DDPM


def get_args():
    parser = argparse.ArgumentParser(
        description="Denoising Diffusion Probabilistic Models"
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        '--num_workers',
        type=int,
        help='number of data loading workers',
        default=8
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        help='directory where snapshot.pt is'
    )
    parser.add_argument('--seed', type=int, help='random seed (default: 0)', default=0)
    parser.add_argument('--device_id', type=int, help='cuda device id (default: 0)', default=0)
    args = parser.parse_args()

    return args
 

def generate_and_save_samples(args, model, sample_dir, num_samples=50000):
    def save_image(kwargs):
        torchvision.utils.save_image(**kwargs)

    for i in range(math.ceil(num_samples/100)):
        batch_size = 100 if i < num_samples // 100 else num_samples % 100
        samples = model.module.sample(batch_size)
        kwargs_list = [{
            "tensor": samples[j],
            "fp": os.path.join(sample_dir, f"{100*i+j:05}.png")
        } for j in range(batch_size)]

        with Pool(args.num_workers) as p:
            p.map(save_image, kwargs_list)


def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "main":
    args = get_args()
    set_seed(args.seed)

    data_dir = None
    data_loader = None
    if args.dataset == "cifar10":
        in_channels = 3
        resolution = 32
        channels = [128, 256, 256, 256]

    elif args.dataset == "celebahq":
        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
        data_dir = "./data/celebahq/data256x256"

    elif args.dataset == "lsun_bedroom":
        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
        data_dir = "./data/lsun_bedroom"

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

            dataset = LSUN(
                "./data/lsun",
                classes=["bedroom_train"],
                transform=v2.Compose([
                    v2.ToImage(),
                    v2.CenterCrop(256),
                    v2.ToDtype(torch.float32, scale=True)
                ]),
            )

            def save_data(data):
                idx, (x, y) = data
                torchvision.utils.save_image(
                    x, 
                    os.path.join(data_dir, f"{idx:05}.png")
                )

            with Pool(args.num_workers) as p:
                p.map(save_data, list(enumerate(dataset)))

    elif args.dataset == "lsun_church":
        in_channels = 3
        resolution = 256
        channels = [128, 128, 256, 256, 512, 512]
    else:
        raise NotImplementedError
    
    device = "cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    model = DDPM(resolution, in_channels, channels).to(device)
    model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

    snapshot_path = os.path.join(args.log_dir, "snapshot.pt")
    snapshot = torch.load(args.snapshot_path, map_location=device)
    model.load_state_dict(snapshot["EMA_MODEL_STATE"])

    sample_dir = os.path.join(args.log_dir, "samples")
    if os.path.exists(sample_dir):
        files = glob(os.path.join(sample_dir, "*.png"))
        if len(files) != 50000:
            raise ValueError("Total number of files must be 50000")
    else:
        os.mkdir(sample_dir)
        generate_and_save_samples(args, model, sample_dir)

    fid = fid.compute_fid(sample_dir, data_dir, dataset_name=args.dataset)

    writer = SummaryWriter(args.log_dir)
    writer.add_hparams(
        {"dataset": args.dataset},
        {"fid": fid}
    )
