import torch
import random
import footsteps
import os
from models import networks
from data import dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12)
args = parser.parse_args()

footsteps.initialize(run_name='registration-augmentation', output_root="results/")
source_dataset = torch.cat([dataset.get_source_dataset(split='train', segmented=False), dataset.get_source_dataset(split='train', segmented=True)[0]], dim=0)

def make_batch():
    idx = range(len(source_dataset))
    imgs_idx = [random.choice(idx) for _ in range(BATCH_SIZE)]

    imgs = [source_dataset[i:i+1] for i in imgs_idx]
    imgs = torch.cat(imgs).cuda().float()
    return imgs
            
if __name__ == '__main__':
    BATCH_SIZE = args.batch_size

    input_shape = source_dataset[:1, :1].size()
    networks.train_two_stage_registration(input_shape, lambda : (make_batch(), make_batch()), 1, 20000, BATCH_SIZE)