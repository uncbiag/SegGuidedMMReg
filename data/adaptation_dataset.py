import os
from data.base_dataset import BaseDataset
import random
import torch
from models import networks
import data
import monai
from data import dataset

class AdaptationDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
    
        self.source_images_wo_segmentation = dataset.get_source_dataset(split='train', segmented=False)
        self.source_images, self.source_images_seg = dataset.get_source_dataset(split='train', segmented=True)
        
        self.As, self.seg_As = self.augment()
        self.Bs = dataset.get_target_dataset(split='train', segmented=False)
        
        self.A_size = len(self.As)  # get the size of dataset A
        self.B_size = len(self.Bs)  # get the size of dataset B

    #raugmnetation by registration
    def augment(self):
        source_registration_network = networks.make_registration_network(self.source_images[:1, :1].size(), include_last_step=True)
        source_registration_network.regis_net.load_state_dict(torch.load(self.opt.reg_net_path))
        source_registration_network.cuda()
        source_registration_network.eval()

        augmented_images, augmented_segmentatitons = [], []

        with torch.no_grad():
            for i in range(self.source_images.shape[0]):
                for j in range(self.source_images_wo_segmentation.shape[0]):
                    
                    imageA, segA = self.source_images[i:i+1], self.source_images_seg[i:i+1]
                    imageB = self.source_images_wo_segmentation[j:j+1]
                    
                    source_registration_network(imageA.cuda(), imageB.cuda())
                    
                    augmented_images.append(source_registration_network.warped_image_A.detach().cpu())
                    augmented_segmentatitons.append(source_registration_network.as_function(segA.cuda().float(), spline_order=0)(source_registration_network.phi_AB_vectorfield).detach().cpu().int())
        
        del source_registration_network
                
        images = torch.cat([self.source_images, torch.cat(augmented_images, dim=0)], dim=0)
        segmentations = torch.cat([self.source_images_seg, torch.cat(augmented_segmentatitons, dim=0)], dim=0)
        
        return images, segmentations  

    def __getitem__(self, index):
        A_index = random.sample(range(0, self.A_size), 1)[0]
        B_index = random.sample(range(0, self.B_size), 1)[0]
        
        A = self.As[A_index]
        B = self.Bs[B_index]
        
        A_seg = self.seg_As[A_index]
    
        return {'A': A, 'B': B, 'A_seg': A_seg}

    def __len__(self):
        return max(self.A_size, self.B_size)
