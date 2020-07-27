import os, sys
import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils
import imageio
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader

configs = {}
configs['width'] = 384
configs['height'] = 256
configs['batch_size'] = 1
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
configs['num_epochs'] = 2

test_dataset = KittiLoader({**configs, 'mode':'test'})
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         )

models_dir = os.path.join(configs['logging_dir'], 'models')
monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
# monocular_nvs_network = torch.load_state_dict(torch.load(os.path.join(models_dir, str(0).zfill(4)+'_snapshot.pt')))
# monocular_nvs_network.eval()
os.makedirs(models_dir, exist_ok=True)


steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)

def move_forward(data, range, frames):
    steps = torch.linspace(0, range, frames)
    start_t_vec = torch.zeros_like(data['t_vecs'])
    novel_views = []
    for i in range(frames):
        end_t_vec = torch.zeros_like(start_t_vec)
        end_t_vec[2] = steps[i]
        novel_view, alphas = monocular_nvs_network(data['input_img'], data['k_mats'], torch.eye(3), end_t_vec)
        novel_views.append(novel_view.item())
    imageio.mimwrite(os.path.join(configs['logging_dir'], str(steps) +'_forward'), novel_views)


print('Demo mode')
for itr, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
    data = {k:v.float().cuda(0) for k,v in data.items()}
    novel_view, alphas = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], data['t_vecs'])
    start_t_vec = torch.zeros_like(data['t_vecs'])
    move_forward(data, 10, 16)
    # # 
    # for v in range(16):
    #     a = (v)/(16-1)
    #     t_vec =  (1-a)*start_t_vec + a*data['t_vecs']

    # print(f'epoch {epoch} iteration {itr} loss {loss.item()}')

    # if(steps % 200 == 0):
    #     torchvision.utils.save_image(novel_view, os.path.join(configs['logging_dir'], str(steps) +'_novel.png'))
    #     torchvision.utils.save_image(data['target_img'], os.path.join(configs['logging_dir'], str(steps) +'_target.png'))
    #     torchvision.utils.save_image(data['input_img'], os.path.join(configs['logging_dir'], str(steps) +'_input.png'))
    # steps += 1


    # torch.save(os.path.join(models_dir, str(epoch).zfill(4)+'_snapshot.pt'), monocular_nvs_network.state_dict())
    #### here you can do tests every epoch
