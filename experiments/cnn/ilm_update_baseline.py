
import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import io
from PIL import Image
from functools import partial
import sys

sys.path.append("/home/air/projects/ILM-VP-main")
from data import prepare_expansive_data, IMAGENETCLASSES, IMAGENETNORMALIZE
from algorithms import generate_label_mapping_by_frequency, get_dist_matrix, label_mapping_base
from tools.misc import gen_folder_name, set_seed
from tools.mapping_visualization import plot_mapping
from models import ExpansiveVisualPrompt
from cfg import *

from copy import deepcopy


class ZOExpansiveVisualPrompt(ExpansiveVisualPrompt):
    def forward(self, x, return_interval=False):
        if return_interval:
            return super().forward(x)
        else:
            return super().forward(x)


def cge(func, params_dict, mask_dict, step_size, base=None, num_samples=10, momentum=0.9, adapt_step=True):
    if base is None:
        base = func(params_dict)
    grads_dict = {}
    velocities = {}

    for key, param in params_dict.items():
        device = param.device  
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten().to(device)
        else:
            mask_flat = torch.ones_like(param).flatten()

        indices = mask_flat.nonzero().flatten()
        p_flat = param.flatten()

        grad_sum = torch.zeros_like(p_flat)

        for _ in range(num_samples):
            perturbations = torch.zeros_like(p_flat)
            perturbations[indices] = torch.randn(indices.size(0), device=device) * step_size
            perturbed_params_dict = deepcopy(params_dict)
            perturbed_params_dict[key] = (p_flat + perturbations).view_as(param)
            forward_loss = func(perturbed_params_dict)

            perturbed_params_dict[key] = (p_flat - perturbations).view_as(param)
            backward_loss = func(perturbed_params_dict)

            grad_estimate = (forward_loss - backward_loss) / (2 * step_size)
            grad_sum[indices] += grad_estimate * perturbations[indices]

        grad = grad_sum / num_samples

        if key not in velocities:
            velocities[key] = torch.zeros_like(grad)
        velocities[key] = momentum * velocities[key] + (1 - momentum) * grad

        grads_dict[key] = velocities[key].view_as(param).to(param.device)

    if adapt_step:
        grad_norm = torch.norm(torch.cat([g.flatten() for g in grads_dict.values()]))
        step_size = min(step_size, 1.0 / grad_norm)

    return grads_dict, step_size


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "instagram"], default="resnet18")
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--dataset',
                   choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn",
                            "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="flowers102")
    p.add_argument('--mapping-interval', type=int, default=1)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--zoo-sample-size', type=int, default=192)
    p.add_argument('--zoo-step-size', type=float, default=5e-3)
    args = p.parse_args()

    # Misc
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    exp = f"cnn/ilm_vp_zo"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))

    # Data
    loaders, configs = prepare_expansive_data(args.dataset, data_path=data_path)
    normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])

    # Network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "instagram":
        from torch import hub

        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network.requires_grad_(False)
    network.eval()

    # Visual Prompt
    visual_prompt = ZOExpansiveVisualPrompt(224, mask=configs['mask'], normalize=normalize).to(device)

    # Make dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Train
    best_acc = 0.
    step_size = args.zoo_step_size
    velocities = {}

    for epoch in tqdm(range(args.epoch)):
        if epoch % args.mapping_interval == 0:
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)

        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Epo {epoch} Training", ncols=100)

        for x, y in tqdm(pbar):
            x, y = x.to(device), y.to(device)

            params_dict = {name: param.clone().detach() for name, param in visual_prompt.named_parameters() if
                           param.requires_grad}
            mask_dict = {name: param for name, param in visual_prompt.named_buffers() if 'mask' in name}


            def loss_func(params):
                state_dict_backup = visual_prompt.state_dict()
                visual_prompt.load_state_dict(params, strict=False)
                loss = F.cross_entropy(label_mapping(network(visual_prompt(x))), y).detach().item()
                visual_prompt.load_state_dict(state_dict_backup)
                return loss


            grads_dict, step_size = cge(loss_func, params_dict, mask_dict, step_size, num_samples=args.zoo_sample_size,
                                        momentum=0.9, adapt_step=True)

            # Update visual_prompt parameters
            with torch.no_grad():
                for name, param in visual_prompt.named_parameters():
                    if param.requires_grad:
                        param.copy_(param - step_size * grads_dict[name])

            with torch.no_grad():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')

            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")

        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        fx0s = []
        ys = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            fx0s.append(fx0)
            pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
        fx0s = torch.cat(fx0s).cpu()
        ys = torch.cat(ys).cpu()
        mapping_matrix = get_dist_matrix(fx0s, ys)

        with io.BytesIO() as buf:
            plot_mapping(mapping_matrix, mapping_sequence, buf, row_names=configs['class_names'],
                         col_names=np.array(IMAGENETCLASSES))
            buf.seek(0)
            im = transforms.ToTensor()(Image.open(buf))
        logger.add_image("mapping-matrix", im, epoch)
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_sequence": mapping_sequence,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
