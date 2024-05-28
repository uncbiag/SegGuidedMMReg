import random
import torch
import os
import monai
from tqdm import tqdm
import numpy as np
import argparse 
from models import networks
from data import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--train_steps', type=int, default=5000)
parser.add_argument('--channel_num', type=int, default=1)
parser.add_argument('--class_num', type=int, default=4)
parser.add_argument('--reg_net_path', type=str, required=True)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
SEG_TRAIN_STEPS = args.train_steps
CHANNEL_NUM = args.channel_num
CLASS_NUM = args.class_num
PATH = args.reg_net_path
OUTPUT_DIR = 'results/segmentation-source'

#load source image datasets
source_images_wo_segmentation = dataset.get_source_dataset(split='train', segmented=False)
source_images_train, source_images_train_seg = dataset.get_source_dataset(split='train', segmented=True)
source_images_val, source_images_val_seg = dataset.get_source_dataset(split='val', segmented=True)

#load pretrained registration network
source_registration_network = networks.make_registration_network(source_images_train[:1, :1].size(), include_last_step=True)
source_registration_network.regis_net.load_state_dict(torch.load(PATH))
source_registration_network.cuda()
source_registration_network.eval()

#augment images w/o segmentations
augmented_images, augmented_segmentatitons = [], []
with torch.no_grad():
    for i in range(source_images_train.shape[0]):
        for j in range(source_images_wo_segmentation.shape[0]):
            
            imageA, segA = source_images_train[i:i+1].cuda(), source_images_train_seg[i:i+1].cuda()
            imageB = source_images_wo_segmentation[j:j+1].cuda()
            
            source_registration_network(imageA, imageB)
            
            augmented_images.append(source_registration_network.warped_image_A.detach().cpu())
            augmented_segmentatitons.append(source_registration_network.as_function(segA.float(), spline_order=0)(source_registration_network.phi_AB_vectorfield).detach().cpu().int())

del source_registration_network

#concatenate augmented images to source image datasets
source_images_train = torch.cat([source_images_train, torch.cat(augmented_images, dim=0)], dim=0)
source_images_train_seg = torch.cat([source_images_train_seg, torch.cat(augmented_segmentatitons, dim=0)], dim=0)

def get_segmentation_batch(batch_size=BATCH_SIZE):
    idx = [random.randint(0, source_images_train.shape[0]-1) for _ in range(batch_size)]
    img = source_images_train[idx]
    seg = source_images_train_seg[idx]
    return img, seg

#create segmentation network
source_segmentation_network = networks.make_segmentation_network(in_channels=CHANNEL_NUM, out_channels=CLASS_NUM)
source_segmentation_network.cuda()
source_segmentation_network.train()

#define losses and optimizer
dice_loss_train = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
dice_loss_val = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(source_segmentation_network.parameters(), lr=5e-4)

best_loss = 1000

#save path results/segmentation-source/seg_net.pt

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# train segmentation network
for i in tqdm(range(SEG_TRAIN_STEPS)):
    img, seg = get_segmentation_batch(batch_size=BATCH_SIZE)
    
    optimizer.zero_grad()
    pred = source_segmentation_network(img.cuda())
    loss = dice_loss_train(pred, seg.cuda())
    loss.backward()
    optimizer.step()
    
    #validation
    if i % 100 == 0:
        source_segmentation_network.eval()
        losses = []
        
        for j in range(source_images_val.shape[0]):
            img, seg = source_images_val[j:j+1], source_images_val_seg[j:j+1]
            with torch.no_grad():
                pred = source_segmentation_network(img.cuda())
            loss = dice_loss_val(pred, seg.cuda())
            losses.append(loss.item())
    
        print('Validation loss: ', np.mean(losses))
        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            torch.save(source_segmentation_network.state_dict(), f'{OUTPUT_DIR}/seg_net.pt')
        source_segmentation_network.train()