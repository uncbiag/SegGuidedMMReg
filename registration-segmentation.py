import torch
import random
import footsteps
import os
from models import networks
from data import dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--source_seg_net_path', type=str, required=True)
parser.add_argument('--target_seg_net_path', type=str, required=True)
parser.add_argument('--channel_num', type=int, default=1)
parser.add_argument('--class_num', type=int, default=4)
args = parser.parse_args()

CHANNEL_NUM = args.channel_num
CLASS_NUM = args.class_num

footsteps.initialize(run_name='registration-segmentation', output_root="results/")

source_dataset_wo_segmentation = dataset.get_source_dataset(split='train', segmented=False)
source_dataset_segs = dataset.get_source_dataset(split='train', segmented=True)[1]

target_dataset = dataset.get_target_dataset(split='train')

source_seg_model = networks.make_segmentation_network(in_channels=CHANNEL_NUM, out_channels=CLASS_NUM)
target_seg_model = networks.make_segmentation_network(in_channels=CHANNEL_NUM, out_channels=CLASS_NUM)

source_seg_model.load_state_dict(torch.load(f'{args.source_seg_net_path}'))
target_seg_model.load_state_dict(torch.load(f'{args.target_seg_net_path}'))

source_seg_model.cuda().eval()
target_seg_model.cuda().eval()

#obtain segmentations with pretrained segmentation networks
source_pred_segs = []
for i in range(source_dataset_wo_segmentation.shape[0]):
    image = source_dataset_wo_segmentation[i:i+1]

    with torch.no_grad():
        pred = source_seg_model(image.cuda())
        pred = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
        source_pred_segs.append(pred.cpu())
del source_seg_model

target_pred_segs = []
for i in range(target_dataset.shape[0]):
    image = target_dataset[i:i+1]

    with torch.no_grad():
        pred = target_seg_model(image.cuda())
        pred = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
        target_pred_segs.append(pred.cpu())

del target_seg_model

source_dataset_segs = torch.cat([source_dataset_segs, torch.cat(source_pred_segs, dim=0)], dim=0)
target_dataset_segs = torch.cat(target_pred_segs, dim=0)
segmentations = torch.cat([source_dataset_segs, target_dataset_segs], dim=0)

def make_batch():
    idx = range(len(segmentations))
    imgs_idx = [random.choice(idx) for _ in range(BATCH_SIZE)]

    imgs = [segmentations[i:i+1] for i in imgs_idx]
    imgs = torch.cat(imgs).cuda().float()
    return imgs
            
if __name__ == '__main__':
    BATCH_SIZE = args.batch_size

    input_shape = segmentations[:1, :1].size()
    networks.train_single_stage_segmentation_registration(input_shape, lambda : (make_batch(), make_batch()), 1, 20000, BATCH_SIZE)