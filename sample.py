import argparse
import os
import math

import torch
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image

from vqvae1 import VQVAE1
from vqvae2 import VQVAE2


def load_model(args):
    ckpt = torch.load(os.path.join('checkpoint', args.checkpoint))

    weights = {}
    for k, v in ckpt.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights[new_k] = v

    if args.vqvae == '1':
        model = VQVAE1(channel=512, n_res_block=0, n_res_channel=32, embed_dim=256, n_embed=8192, stride=6)
    else:
        model = VQVAE2()

    model.load_state_dict(weights)
    model = model.to(args.device)
    model.eval()
    return model


def load_img(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img_path = os.path.join('test_sample', args.input_img)
    img_ = Image.open(img_path)
    img = transform(img_).unsqueeze(0)
    img = img.to(args.device)
    return img


def img2code(model, img):
    '''Convert a batch of img to code, only for VQ-VAE1
    Args:
        model: The tokenizer model.
        img: [b, c, h, w]
    '''
    with torch.no_grad():
        quant_t1, _, id_t1 = model.encode(img)
    return id_t1.view(img.shape[0], -1)


def code2img(model, code):
    '''Convert a batch of code to imgs, only for VQ-VAE1
    Args:
        model: ...
        code: [b, h, w] or [b, h*w] LongTensor
    '''
    if len(code.shape) == 2:
        s = int(math.sqrt(len(code.view(-1))) + 1e-5)
        code = code.view(code.shape[0], s, s)
    with torch.no_grad():
        out = model.decode_code(code)
        # out = out * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor(
        #     [0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1)
        out = out * torch.tensor([0.5, 0.5, 0.5], device=out.device).view(1, -1, 1, 1) + torch.tensor(
            [0.5, 0.5, 0.5], device=out.device).view(1, -1, 1, 1)
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vqvae', type=str, default='1')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('input_img', type=str)
    parser.add_argument('--output_img', type=str, default='output.jpg')

    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("args\n", args)

    model = load_model(args)
    img = load_img(args)

    code = img2code(model=model, img=img)
    print("code:\n", code)
    output = code2img(model=model, code=code)
    save_image(output, args.output_img)
    # save_image(output, args.output_img, normalize=True, range=(-1, 1))

