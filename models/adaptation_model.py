import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import monai


class AdaptationModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_source, lambda_target, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_source: A -> B; G_target: B -> A.
        Discriminators: D_source: G_source(A) vs. B; D_target: G_target(B) vs. A.
        Forward cycle loss:  lambda_source * ||G_target(G_source(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_target * ||G_source(G_target(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_source(B) - B|| * lambda_target + ||G_target(A) - A|| * lambda_source) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_source', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_target', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_seg', type=float, default=5, help='weight for segmentation loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_source', 'G_source', 'cycle_source', 'idt_source', 'D_target', 'G_target', 'cycle_target', 'idt_target', 'real_target_dice', 'fake_source_dice', 'fake_target_dice', 'rec_source_dice', 'total_seg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_source = ['real_source', 'fake_target', 'rec_source']
        visual_names_target = ['real_target', 'fake_source', 'rec_target']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_target=G_source(B) ad idt_source=G_source(B)
            visual_names_source.append('idt_target')
            visual_names_target.append('idt_source')

        self.visual_names = visual_names_source + visual_names_target  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_source', 'G_target', 'D_source', 'D_target', 'S_target']
        else:  # during test time, only load Gs
            self.model_names = ['G_source', 'G_target']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_source (G), G_target (F), D_source (D_Y), D_target (D_X)
        self.netG_source = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_target = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.netS_source = networks.make_segmentation_network(opt.input_nc, opt.seg_class_num)
        self.netS_target = networks.make_segmentation_network(opt.input_nc, opt.seg_class_num)
        
        #initialize segmentation networks with same weight
        self.netS_source.load_state_dict(torch.load(f'{self.opt.seg_net_path}'))
        self.netS_target.load_state_dict(torch.load(f'{self.opt.seg_net_path}'))
        
        self.netS_source.to(self.device).eval()
        self.netS_target.to(self.device).train()
        
        #data parallel
        self.netS_source = torch.nn.DataParallel(self.netS_source, device_ids=self.gpu_ids)
        self.netS_target = torch.nn.DataParallel(self.netS_target, device_ids=self.gpu_ids)
        
        if self.isTrain:  # define discriminators
            self.netD_source = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_target = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_source_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_target_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.dice_loss = monai.losses.DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
            self.dice_loss2 = monai.losses.DiceLoss(to_onehot_y=False, softmax=False, include_background=True)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_source.parameters(), self.netG_target.parameters(), self.netS_target.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_source.parameters(), self.netD_target.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        
        self.real_source = input['A'].to(self.device)
        self.real_target = input['B'].to(self.device)
        self.seg_source = input['A_seg'].to(self.device)
            
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_target = self.netG_source(self.real_source)  # G_source(A)
        self.rec_source  = self.netG_target(self.fake_target)  # G_target(G_source(A))
        
        self.fake_source = self.netG_target(self.real_target)  # G_target(B)
        self.rec_target  = self.netG_source(self.fake_source)  # G_source(G_target(B))
        
        self.fake_seg_target = self.netS_target(self.fake_target) # S_target(G_source(A))
        self.rec_seg_source  = self.netS_source(self.rec_source)  # S_source(G_target(G_source(A)))
           
        self.real_seg_target = self.netS_target(self.real_target) # S_target(B)
        self.fake_seg_source = self.netS_source(self.fake_source) # S_source(G_target(B))
        self.rec_seg_target  = self.netS_target(self.rec_target)  # S_target(G_source(G_target(B)))
        
    def backward_D_targetasic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_source(self):
        """Calculate GAN loss for discriminator D_source"""
        fake_target = self.fake_target_pool.query(self.fake_target)
        self.loss_D_source = self.backward_D_targetasic(self.netD_source, self.real_target, fake_target)

    def backward_D_target(self):
        """Calculate GAN loss for discriminator D_target"""
        fake_source = self.fake_source_pool.query(self.fake_source)
        self.loss_D_target = self.backward_D_targetasic(self.netD_target, self.real_source, fake_source)

    def dice_softmax(self, mask1, mask2):
        mask1 = torch.softmax(mask1, dim=1)
        mask2 = torch.softmax(mask2, dim=1)
        
        return self.dice_loss2(mask1, mask2)
        
    def backward_G(self):
        """Calculate the loss for generators G_source and G_target"""
        lambda_idt = self.opt.lambda_identity
        lambda_source = self.opt.lambda_source
        lambda_target = self.opt.lambda_target
        lambda_seg = self.opt.lambda_seg
        # Identity loss
        if lambda_idt > 0:
            # G_source should be identity if real_target is fed: ||G_source(B) - B||
            self.idt_source = self.netG_source(self.real_target)
            self.loss_idt_source = self.criterionIdt(self.idt_source, self.real_target) * lambda_target * lambda_idt
            # G_target should be identity if real_source is fed: ||G_target(A) - A||
            self.idt_target = self.netG_target(self.real_source)
            self.loss_idt_target = self.criterionIdt(self.idt_target, self.real_source) * lambda_source * lambda_idt
        else:
            self.loss_idt_source = 0
            self.loss_idt_target = 0

        # GAN loss D_source(G_source(A))
        self.loss_G_source = self.criterionGAN(self.netD_source(self.fake_target), True)
        # GAN loss D_target(G_target(B))
        self.loss_G_target = self.criterionGAN(self.netD_target(self.fake_source), True)
        # Forward cycle loss || G_target(G_source(A)) - A||
        self.loss_cycle_source = self.criterionCycle(self.rec_source, self.real_source) * lambda_source
        # Backward cycle loss || G_source(G_target(B)) - B||
        self.loss_cycle_target = self.criterionCycle(self.rec_target, self.real_target) * lambda_target
        
        # Segmentation Losses
        self.loss_fake_target_dice = self.dice_loss(self.fake_seg_target, self.seg_source)
        self.loss_rec_source_dice  = self.dice_loss(self.rec_seg_source, self.seg_source)
        
        self.loss_real_target_dice = self.dice_softmax(self.real_seg_target, self.fake_seg_source)
        self.loss_fake_source_dice = self.dice_softmax(self.fake_seg_source, self.rec_seg_target)  

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_source + self.loss_G_target + self.loss_cycle_source + self.loss_cycle_target + self.loss_idt_source + self.loss_idt_target
        self.loss_total_seg = (self.loss_fake_target_dice + self.loss_rec_source_dice + self.loss_real_target_dice + self.loss_fake_source_dice) * lambda_seg 
        
        total_loss = self.loss_G + self.loss_total_seg
        total_loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_source and G_target
        self.set_requires_grad([self.netD_source, self.netD_target], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_source and G_target's gradients to zero
        self.backward_G()             # calculate gradients for G_source and G_target
        self.optimizer_G.step()       # update G_source and G_target's weights
        # D_source and D_target
        self.set_requires_grad([self.netD_source, self.netD_target], True)
        self.optimizer_D.zero_grad()   # set D_source and D_target's gradients to zero
        self.backward_D_source()      # calculate gradients for D_source
        self.backward_D_target()      # calculate graidents for D_target
        self.optimizer_D.step()  # update D_source and D_target's weights
