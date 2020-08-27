import os, sys
import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils
from .gan_opts import arg_parser
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader
from monocular.src import Trainer
from monocular.src import MultiscaleDiscriminator
gan_opts = arg_parser.parse_args()

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

## Dataset related settings
is_teddy = True
if is_teddy:
    # configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
    configs['dataset_root'] = '/data/teddy/KITTI_Odometry/dataset'
    configs['logging_dir'] = '/habtegebrialdata/monocular_nvs/experiment_logs/exp_1_with_bn_new_alpha_comp'
else:
    configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
    configs['logging_dir'] = '/home/anwar/data/experiments/exp5'

configs['mode'] = 'train'
configs['max_baseline'] = 2
configs['num_epochs'] = 10

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
gen_optimizer = torch.optim.Adam(monocular_nvs_network.parameters(), lr=gan_opts['lr_gen'], betas=(0.9, 0.999))
gen_optimizer.zero_grad()


discriminator = MultiscaleDiscriminator(gan_opts).cuda(0)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan_opts['lr_disc'], betas=(0.9, 0.999))
disc_optimizer.zero_grad()

trainer = Trainer(gan_opts).cuda(0)
trainer.initialise(monocular_nvs_network, disc_optimizer)


models_dir = os.path.join(configs['logging_dir'], 'models')
os.makedirs(models_dir, exist_ok=True)
steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)

for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        data = {k:v.float().cuda(0) for k,v in data.items()}
        gen_losses = trainer(data, mode='generator')
        gen_l = sum([v for k,v in gen_losses.items()]).mean()
        gen_l.backward()
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        disc_losses = trainer(data, mode='discriminator')
        disc_l = sum([v for k,v in disc_losses.items()]).mean()
        disc_l.backward()
        disc_optimizer.step()
        disc_optimizer.zero_grad()
        novel_view = (trainer.fake + 1.0)/2.0
        gen_print = {k:v.item() for k,v in gen_losses.items()}
        disc_print = {k:v.item() for k,v in disc_losses.items()}
        print(f'epoch {epoch} iteration {itr}     generator  loss {gen_print}')
        print(f'epoch {epoch} iteration {itr}  discriminator loss {disc_print}')
        if(steps % 200 == 0):
            novel_view = novel_view.data[:, [2,1,0], :, :].cpu()
            target = data['target_img'].data[:, [2,1,0], :, :].cpu()
            input_img = data['input_img'].data[:, [2,1,0], :, :].cpu()
            torchvision.utils.save_image(novel_view, os.path.join(configs['logging_dir'], str(steps) +'_novel.png'))
            torchvision.utils.save_image(target, os.path.join(configs['logging_dir'], str(steps) +'_target.png'))
            torchvision.utils.save_image(input_img, os.path.join(configs['logging_dir'], str(steps) +'_input.png'))
        steps += 1

    torch.save(monocular_nvs_network.state_dict(), os.path.join(models_dir, str(epoch).zfill(4)+'gen_snapshot.pt'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, str(epoch).zfill(4)+'disc_snapshot.pt'))
    #### here you can do tests every epoch
