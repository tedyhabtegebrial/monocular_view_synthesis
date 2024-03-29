import os, sys
import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader, RealEstateLoader

configs = {}
configs['width'] = 384
configs['height'] = 256
configs['batch_size'] = 4
configs['num_planes'] = 32
configs['near_plane'] = 5
configs['far_plane'] = 20000
configs['encoder_features'] = 32
configs['encoder_ouput_features'] = 64
configs['input_channels'] = 3
configs['out_put_channels'] = 3
configs['num_features'] = 16
configs['occlusion_levels'] = 3

## Dataset related settings
is_teddy = False
if is_teddy:
    # configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
    configs['dataset_root'] = '/data/teddy/KITTI_Odometry/dataset'
    configs['logging_dir'] = '/habtegebrialdata/monocular_nvs/experiment_logs/exp_1_with_bn_new_alpha_comp'
else:
    # configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
    configs['dataset_root'] = '/home5/anwar/data/realestate10k/'
    configs['logging_dir'] = '/home5/anwar/data/experiments/exp_ReSt_GAN_LOSS'

configs['mode'] = 'train'
configs['max_baseline'] = 3
configs['num_epochs'] = 10

train_dataset = RealEstateLoader(configs)
# train_dataset = KittiLoader(configs)
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=configs['batch_size'],
                         shuffle=True,
                         num_workers=max(1, configs['batch_size']//2),
                         )
# test_dataset = KittiLoader({**configs, 'mode':'test'})
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=1,
#                          shuffle=False,
#                          )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
optimizer = torch.optim.Adam(monocular_nvs_network.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer.zero_grad()
models_dir = os.path.join(configs['logging_dir'], 'models')
os.makedirs(models_dir, exist_ok=True)
steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)

for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        data = {k:v.float().cuda(0) for k,v in data.items()}
        novel_view, alphas = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], data['t_vecs'])
        loss = F.l1_loss(novel_view, data['target_img'])
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(monocular_nvs_network.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        print(f'epoch {epoch} iteration {itr} loss {loss.item()*255}')
        if(steps % 50 == 0):
            #  novel_view.data
            novel_view = novel_view.data[:, [2,1,0], :, :].cpu()
            target = data['target_img'].data[:, [2,1,0], :, :].cpu()
            input_img = data['input_img'].data[:, [2,1,0], :, :].cpu()
            torchvision.utils.save_image(novel_view, os.path.join(configs['logging_dir'], str(steps) +'_novel.png'))
            torchvision.utils.save_image(target, os.path.join(configs['logging_dir'], str(steps) +'_target.png'))
            torchvision.utils.save_image(input_img, os.path.join(configs['logging_dir'], str(steps) +'_input.png'))
        steps += 1

    torch.save(monocular_nvs_network.state_dict(), os.path.join(models_dir, str(epoch).zfill(4)+'_snapshot.pt'))
    #### here you can do tests every epoch
