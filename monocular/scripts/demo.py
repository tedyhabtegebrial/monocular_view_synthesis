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
monocular_nvs_network.eval()
os.makedirs(models_dir, exist_ok=True)


steps = 0
print('Logging info: ', configs['logging_dir'])

# torch.autograd.set_detect_anomaly(True)

def create_images(data, t_vecs, direction):
    novel_views = []
    for i in range(len(t_vecs)):
        with torch.no_grad():
            novel_view = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], -t_vecs[i])
        novel_view = novel_view.squeeze().permute(1, 2, 0)
        novel_views.append(novel_view.cpu().detach())
    imageio.mimwrite(os.path.join(configs['logging_dir'], direction + '_demo.gif'), novel_views)


# Create [frames, 3, 1] vectors in the linear sampling of [0, m_range]
def move_forward(data, m_range, frames):
    steps = torch.linspace(0, m_range, frames).reshape((frames, 1, 1))
    end_t_vec = torch.cat([torch.zeros((frames, 2, 1)), steps], dim=1).cuda(0)
    print(end_t_vec.shape)
    create_images(data, end_t_vec, 'forward')

# Create [frames, 3, 1] vectors in the linear sampling of [0, m_range]
def move_horizontal(data, m_range, frames):
    steps = torch.linspace(-m_range/2, m_range/2, frames).reshape((frames, 1, 1))
    end_t_vec = torch.cat([steps, torch.zeros((frames, 2, 1))], dim=1).cuda(0)
    create_images(data, end_t_vec, 'horizontal')

# Create [frames, 3, 1] vectors in the linear sampling of [0, m_range]
def move_cicular(data, m_range, frames):
    x_sort = torch.linspace(0, m_range, frames//4 + 1).reshape((frames//4 + 1, 1, 1))
    y_rev = torch.sqrt(m_range**2 - x_sort**2)
    y_sort = torch.sort(y_rev, dim=0)[0]
    x_rev = torch.sort(x_sort, dim=0, descending=True)[0]
    x_neg_rev = -x_sort[1:]
    y_neg_rev = -y_sort[1:]
    x_neg_sort = torch.sort(x_neg_rev, dim=0)[0]
    y_neg_sort = torch.sort(y_neg_rev, dim=0)[0]
    x = torch.cat([x_neg_sort, x_sort, x_rev[1:], x_neg_rev[:-1]], dim=0)
    y = torch.cat([y_sort, y_rev, y_neg_rev[1:], y_neg_sort[:-1]], dim=0)

    end_t_vec = torch.cat([x, y, torch.zeros_like(x)], dim=1).cuda(0)
    create_images(data, end_t_vec, 'circular')


print('Demo mode')
for itr, data in enumerate(test_loader):
    data = {k:v.float().cuda(0) for k,v in data.items()}
    # novel_view, alphas = monocular_nvs_network(data['input_img'], data['k_mats'], data['r_mats'], data['t_vecs'])
    # start_t_vec = torch.zeros_like(data['t_vecs'])
    move_forward(data, 10, 16)
    move_horizontal(data, 10, 16)
    move_cicular(data, 10, 16)
    print('moved forward')

    break
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
