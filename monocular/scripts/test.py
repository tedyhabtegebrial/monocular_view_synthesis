import os, sys
import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader
from skimage.metrics import structural_similarity as ssim 

configs = {}
configs['width'] = 384
configs['height'] = 256
configs['batch_size'] = 4
configs['num_planes'] = 64
configs['near_plane'] = 2.5
configs['far_plane'] = 20000
configs['encoder_features'] = 32
configs['encoder_ouput_features'] = 64
configs['input_channels'] = 3
configs['out_put_channels'] = 3

## Dataset related settings
configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
configs['logging_dir'] = '/home/anwar/data/experiments/exp3'
configs['mode'] = 'train'
configs['max_baseline'] = 5
configs['num_epochs'] = 10

# train_dataset = KittiLoader(configs)
# train_loader = DataLoader(dataset=train_dataset,
#                          batch_size=configs['batch_size'],
#                          shuffle=True,
#                          num_workers=max(1, configs['batch_size']//2),
#                          )
test_dataset = KittiLoader({**configs, 'mode':'test'})
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
models_dir = os.path.join(configs['logging_dir'], 'models')
os.makedirs(models_dir, exist_ok=True)
steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)
mses = 0
ssis = 0
l1s = 0

for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        data = {k:v.float().cuda(0) for k,v in data.items()}
        with torch.no_grad():
            novel_view, alphas = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], data['t_vecs'])
        mse = F.mse_loss(novel_view, data['target_img'])
        mses += mse
        ssi = ssim(novel_view, data['target_img'])
        ssis += ssi
        l1 = F.l1_loss(novel_view, data['target_img'])
        l1s += l1

        print(f'epoch {epoch} iteration {itr} \n MSE: {mse.item()} \n SSI: {ssi.item()} \n L1: {l1.item()}')

        # if(steps % 200 == 0):
        #     #  novel_view.data
        #     novel_view = novel_view.data[:, [2,1,0], :, :].cpu()
        #     target = data['target_img'].data[:, [2,1,0], :, :].cpu()
        #     input_img = data['input_img'].data[:, [2,1,0], :, :].cpu()
        #     torchvision.utils.save_image(novel_view, os.path.join(configs['logging_dir'], str(steps) +'_novel.png'))
        #     torchvision.utils.save_image(target, os.path.join(configs['logging_dir'], str(steps) +'_target.png'))
        #     torchvision.utils.save_image(input_img, os.path.join(configs['logging_dir'], str(steps) +'_input.png'))
        steps += 1
    print(f'average loss so far: \n MSE: {mse.item()} \n SSI: {ssi.item()} \n L1: {l1.item()}')

    #### here you can do tests every epoch
