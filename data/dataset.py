import torch

def split_dataset(file, split='train'):
    data = torch.load(file, map_location=torch.device('cpu'))
    
    train_index = int(len(data) * 0.75)
    valid_index = int(len(data) * 0.85)
    
    if split == 'all':
        return data
    elif split == 'train':
        return data[:train_index]
    elif split == 'val':
        return data[train_index:valid_index]
    elif split == 'test':
        return data[valid_index:]
    else:
        raise ValueError('Invalid split type')

#image with segmentation, segmentation, image without segmentation
def get_source_dataset(split='train', segmented=True):
    if segmented:
        return split_dataset('data/source_dataset.pt', split), split_dataset('data/source_dataset_seg.pt', split)
    else:
        return split_dataset('data/source_dataset_wo_seg.pt', split)
    
def get_target_dataset(split='train', segmented=False):
    if segmented:
        return split_dataset('data/target_dataset.pt', split), split_dataset('data/target_dataset_seg.pt', split)
    else:
        return split_dataset('data/target_dataset_wo_seg.pt', split)