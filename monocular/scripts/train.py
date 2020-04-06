import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .monocular.src import StereoMagnification
from .monocular.src import KittiLoader

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
configs['dataset_root'] = '/data/Datasets/KittiOdometry/dataset'
configs['mode'] = 'train'
configs['max_baseline'] = 5

train_dataset = KittiLoader(configs)
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=configs['batch_size'],
                         shuffle=True,
                         num_workers=max(1, configs['batch_size']//2),
                         )
test_dataset = KittiLoader({**configs, 'mode':'test'})
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
optimizer = torch.optim.Adam(monocular_nvs_network.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer.zero_grad()
for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
    data = {k:v.float().cuda(0) for k,v in data.items()}
    novel_view = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], data['t_vecs'])
    loss = F.l1_loss(novel_view, data['target_im'])
    loss.backward()
    optimizer.step()
    print(f'iteration {itr} loss {loss.item()}')
