CHANNELS=1
SEG_CLASSES=4
#train registration model for source dataset augmentation
python registration-augmentation.py
#train segmentation model for source dataset with augmentation
python segment-w-augmentation.py --reg_net_path=./results/registration-augmentation/2nd_step/Step_2_final.trch --channel_num=$CHANNELS --class_num=$SEG_CLASSES
#run domain adaptation for target dataset
python domain-adaptation.py --reg_net_path=./results/registration-augmentation/2nd_step/Step_2_final.trch --seg_net_path=./results/segmentation-source/seg_net.pt --input_nc=$CHANNELS --seg_class_num=$SEG_CLASSES
#train registration network with predicted segmentations
python registration-segmentation.py --source_seg_net_path=./results/segmentation-source/seg_net.pt --target_seg_net_path=./results/domain-adaptation/latest_net_S_target.pth --channel_num=$CHANNELS --class_num=$SEG_CLASSES