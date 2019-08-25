import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import PIL 
import cv2 
import pickle 
import tqdm 
import datetime 
path_to_data = "raven_data/data_a_ab_c_d_e_with_mask_570x900_py3.pkl"
with open(path_to_data, 'rb') as file: 
    raven_data = pickle.load(file) 

save_image_base_dir = "results/" + str(datetime.datetime.now())
import os 
os.mkdir(save_image_base_dir)
# raven_data['a_mask'] = np.zeros((570, 900), dtype=np.uint8) 
# raven_data['a_mask'][285:285+220, 460:470+390] = 255 
# raven_data['ab_mask'] = np.zeros((570, 900), dtype=np.uint8) 
# raven_data['ab_mask'][285:,460:] = 255 
# raven_data['b_mask'] = raven_data['ab_mask'].copy() 

generated_image_idx = 0 

import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default=0, help='manual seed')
# parser.add_argument('--image', type=str)
# parser.add_argument('--mask', type=str, default='')
# parser.add_argument('--output', type=str, default='output.png')
# parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0) # default means the latest iteration 

args = parser.parse_args()
config = get_config(args.config)
# CUDA configuration
cuda = config['cuda']
if torch.cuda.device_count() > 0: 
    cuda = True # memory problem 

if not args.checkpoint_path:
    args.checkpoint_path = os.path.join('checkpoints',
                                    config['dataset_name'],
                                    config['mask_type'] + '_' + config['expname'])

dataset_name = args.checkpoint_path.split("/")[1] 

device_ids = config['gpu_ids']
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    device_ids = list(range(len(device_ids)))
    device_ids = [] 
    # config['gpu_ids'] = device_ids
    cudnn.benchmark = True
print("Arguments: {}".format(args))
print("Use cuda: {}, use gpu_ids: {}".format(cuda, device_ids))





# Set random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
print("Random seed: {}".format(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed_all(args.seed)
# print("Configuration: {}".format(config))






# Define the trainer
netG = Generator(config['netG'], cuda, device_ids)
# Resume weight
# if cuda: 
#     netG.cuda()
last_model_name = get_model_list(args.checkpoint_path, "gen", iteration=args.iter)
print("loading model from here --------------> {}".format(last_model_name))
# if not cuda:
netG.load_state_dict(torch.load(last_model_name, map_location='cpu'))
# else: 
#     netG.load_state_dict(torch.load(last_model_name))
last_model_name = "/home/ubuntu/generative-inpainting-model/checkpoints/imagenet/hole_benchmark/gen_00165000.pt"
model_iteration = int(last_model_name[-11:-3])
print("Resume from {} at iteration {}".format(args.checkpoint_path, model_iteration))

if cuda:
    # netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = netG.to(device)


def get_generated_image(category, idx): # input 'a', '3'; mask is automatically inferred; 
    return _get_generated_image(raven_data[category][idx][0,:,:], raven_data[category+"_mask"]) 

def _get_generated_image(x, mask=None,): 
    # global generated_image_idx 
    global netG 
    if mask is None: 
        mask = np.zeros(x.shape, dtype=np.uint8)

    if len(x.shape) == 2:
        x = PIL.Image.fromarray(np.stack((x,)*3, axis=-1)) 
    elif len(x.shape) == 3 and x.shape[-1] == 3: 
        x = PIL.Image.fromarray(x)
    else: 
        print(x.shape)
        print("dim error")
        import sys 
        sys.exit(0)
    mask = PIL.Image.fromarray(mask) 
    x = transforms.Resize(config['image_shape'][:-1])(x)

    x = transforms.CenterCrop(config['image_shape'][:-1])(x)
    mask = transforms.Resize(config['image_shape'][:-1])(mask)
    mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
    x = transforms.ToTensor()(x)
    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
    x = normalize(x)
    x = x * (1. - mask)
    x = x.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)



    if cuda:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.cuda()
        mask = mask.cuda()


    # Inference
    x1, x2, offset_flow, feature = netG(x, mask)
    inpainted_result = x2 * mask + x * (1. - mask)
    if cuda:
        inpainted_result = inpainted_result.cpu()
    np_inpainted_result = np.rollaxis(np.uint8(np.squeeze(inpainted_result.detach().numpy(), axis=0)), 0,3)
    assert np_inpainted_result.shape == (256, 256, 3) 
    # return 255-np_inpainted_result 

    # vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
    # from PIL import Image
    grid = vutils.make_grid(inpainted_result)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


    return ndarr, feature
    # print("Saved the inpainted result to {}".format(args.output))
    # if args.flow:
    #     vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
    #     print("Saved offset flow to {}".format(args.flow))

def get_feature(img): 
    _, feature = _get_generated_image(img)
    if cuda: 
        feature = feature.cpu()
    feature = feature.detach().numpy() 
    return np.squeeze(feature, 0)  # torch.Size([1, 128, 64, 64]) => [128, 64, 64]

def get_answer(category, idx): 
    # generated_img, _
    global generated_image_idx
    height, width = raven_data[category][idx][0,:,:].shape 
    assert height == 570 and width == 900 
    generated_img, generated_img_feature = get_generated_image(category, idx)
    # save image to result folder
    im = PIL.Image.fromarray(generated_img)
    im.save(os.path.join(save_image_base_dir, str(generated_image_idx)+".png"))
    generated_image_idx += 1 
    resized_img = cv2.resize(generated_img, (width, height)) 
    if category in ['a', 'ab', 'b']:
        lower_right_img = resized_img[height//2:, width//2:,:]  
    elif category in ['c', 'd', 'e']: 
        lower_right_img = resized_img[height//2:, width//2:,:]  

    answer_imgs = raven_data[category][idx][:,height//2:, width//2:] # shape == (6, _, _) 
    features = [] 
    features.append(get_feature(lower_right_img)) 
    for i in range(answer_imgs.shape[0]):
        x = answer_imgs[i,:,:] 
        x = np.stack((x,)*3, axis=-1)
        features.append(get_feature(x))
    if category in ['a', 'ab', 'b']:
        assert len(features) == 1+6
    elif category in ['c', 'd', 'e']: 
        assert len(features) == 1+8 
    dists = [] 
    for i in range(answer_imgs.shape[0]): 
        dist = np.sum(np.square((features[0] - features[i+1])))
        dists.append(dist)
    # print(dists)
    closest_idx = np.argmin(dists)+1
    return closest_idx 



if __name__ == '__main__':

    # img, feature = get_generated_image('ab', 1) 
    # print(feature.shape)
    # print(type(feature)) 
    # print(type(feature.detach().numpy()[0,0,0,0])) 

    # plt.imshow(img) 
    # plt.show()
    # print(get_feature(raven_data['a'][0][0,:,:]).shape)
    from answers import Answers
    # sets = ['a', 'ab', 'b', 'c', 'd', 'e']
    sets = ['a', 'ab', 'b', 'c', 'd', 'e']
    ans = []
    for j, s in enumerate(sets): 
        print("Dealing with {} set".format(s))
        for i in tqdm.tqdm(range(12)):
            ans.append(get_answer(s,i))
    # print(get_answer('a', 0))
    correct_ans = Answers().a + Answers().ab + Answers().b + Answers().c + Answers().d + Answers().e 

    comparison = np.array(correct_ans) == np.array(ans) # boolean comparison 
    correct_cnt = np.sum(comparison) 

    correct_set_cnt = [] 
    for i in range(0, 12*len(sets), 12): 
        correct_set_cnt.append(np.sum(comparison[i:i+12]))

    print(correct_cnt, "------------", correct_cnt/len(correct_ans))
    print(correct_set_cnt)
    with open("./results/answers.txt", 'a') as file: # results folder is there, answers.txt will be created if needed 
        file.write(str(datetime.datetime.now()) + " " + dataset_name + " " + str(model_iteration)+"\n")
        file.write(','.join(map(str, ans))+"\n")
        file.write(','.join(map(str, correct_set_cnt))+"\n") 
        file.write('\n') 
# random seed 698
# 5.27
# 4566
















# def main():
#     try:  # for unexpected error logging
#         with torch.no_grad():   # enter no grad context

#             if is_image_file(args.image):
#                 if args.mask and is_image_file(args.mask):
#                     # Test a single masked image with a given mask
#                     # x = default_loader(args.image)
#                     x = np.stack((raven_data['a'][5][0,:,:],)*3, axis=-1)
#                     x = PIL.Image.fromarray(x) 
                    
#                     # mask = default_loader(args.mask)
#                     mask = PIL.Image.fromarray(raven_data['a_mask'])

#                     x = transforms.Resize(config['image_shape'][:-1])(x)
#                     x = transforms.CenterCrop(config['image_shape'][:-1])(x)
#                     mask = transforms.Resize(config['image_shape'][:-1])(mask)
#                     mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
#                     x = transforms.ToTensor()(x)
#                     mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
#                     x = normalize(x)
#                     x = x * (1. - mask)
#                     x = x.unsqueeze(dim=0)
#                     mask = mask.unsqueeze(dim=0)
#                 elif args.mask:
#                     raise TypeError("{} is not an image file.".format(args.mask))
#                 else:
#                     # Test a single ground-truth image with a random mask
#                     ground_truth = default_loader(args.image)
#                     ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
#                     ground_truth = transforms.CenterCrop(config['image_shape'][:-1])(ground_truth)
#                     ground_truth = transforms.ToTensor()(ground_truth)
#                     ground_truth = normalize(ground_truth)
#                     ground_truth = ground_truth.unsqueeze(dim=0)
#                     bboxes = random_bbox(config, batch_size=ground_truth.size(0))
#                     x, mask = mask_image(ground_truth, bboxes, config)

#                 # Set checkpoint path
#                 if not args.checkpoint_path:
#                     checkpoint_path = os.path.join('checkpoints',
#                                                    config['dataset_name'],
#                                                    config['mask_type'] + '_' + config['expname'])
#                 else:
#                     checkpoint_path = args.checkpoint_path 

#                 # Define the trainer
#                 netG = Generator(config['netG'], cuda, device_ids)
#                 # Resume weight

#                 last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
#                 if not cuda:
#                     netG.load_state_dict(torch.load(last_model_name, map_location='cpu'))
#                 else: 
#                     netG.load_state_dict(torch.load(last_model_name))
#                 model_iteration = int(last_model_name[-11:-3])
#                 print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

#                 if cuda:
#                     netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
#                     x = x.cuda()
#                     mask = mask.cuda()

#                 # Inference
#                 x1, x2, offset_flow = netG(x, mask)
#                 inpainted_result = x2 * mask + x * (1. - mask)
#                 print(inpainted_result.shape) 
#                 print(type(np.uint8(inpainted_result.numpy())[0,0,0,0]))
#                 vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
#                 print("Saved the inpainted result to {}".format(args.output))
#                 if args.flow:
#                     vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
#                     print("Saved offset flow to {}".format(args.flow))
#             else:
#                 raise TypeError("{} is not an image file.".format)
#         # exit no grad context
#     except Exception as e:  # for unexpected error logging
#         print("Error: {}".format(e))
#         raise e
