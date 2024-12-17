import os
import argparse
import torch
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
import Utils
import warnings

warnings.filterwarnings("ignore")



def get_defense(args):
    if args.defense_method == 'anti-diffusion':
        from Defense import antidiffusion
        defense = antidiffusion.defense
    else:
        print("Protection {} not supported.".format(args.protection_method))
        exit(-1)
    return defense

def main(args):
    seed_everything(args.seed)
    

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = StableDiffusionPipeline.from_pretrained(
        args.diffusion_path, safety_checker=None).to(device)
    images_dataset = Utils.ImageData(args.images_root, args.image_size)
    images_loader = DataLoader(images_dataset, batch_size=args.batch_size)
    
    if os.path.isdir(args.save_dir):
        pass
    else:
        os.makedirs(args.save_dir)
    print("output dir is " + args.save_dir)

    defense = get_defense(args)
    
    print("Using defense method "+args.defense_method)

    for ind, [image_names, images] in enumerate(images_loader):
        images = images.to(model.device)
        protected_images, loss = defense(images, model, args)
        Utils.save_images(protected_images, image_names, args)
        # print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Global seed')
    parser.add_argument('--save_dir', default="Outputs", type=str,
                        help='Where to save the examples, and other results')
    parser.add_argument('--images_root', default="Dataset/CelebA-Test", type=str,
                        help='The clean images root directory')
    parser.add_argument("--image_size", default=512, type=int,
                        help="The image size when processing")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="The size of batch when processing")
    parser.add_argument("--diffusion_path",
                        default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--defense_method", default="anti-diffusion")
    parser.add_argument("--pgd_alpha", default=2e-3, type=float)
    parser.add_argument("--pgd_eps", default=5e-2, type=float)
    parser.add_argument("--pgd_itrs", default=10, type=int)
    parser.add_argument("--epoches", default=5, type=int)
    parser.add_argument("--learning_rate", default=5e-7, type=float)
    parser.add_argument("--tunning_steps",default=20,type=int)
    parser.add_argument("--attention_size",default=16,type=int)
    args = parser.parse_args()
    main(args)
