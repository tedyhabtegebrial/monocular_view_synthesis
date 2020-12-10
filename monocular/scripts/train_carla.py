import os
import sys
import tqdm
import time
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils as utils
from .gan_opts import arg_parser
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader
from monocular.src import RealEstateLoader
from monocular.src import Trainer
from monocular.src import MultiscaleDiscriminator
from monocular.src import SingleViewNetwork_DFKI
from monocular.src import CarlaLoader
from monocular.src import SpacesLoader
from tensorboardX import SummaryWriter

gan_opts = arg_parser.parse_args()

configs = {}
configs['width'] = 384
configs['height'] = 256
configs['batch_size'] = 6
configs['num_planes'] = 32
configs['near_plane'] = 5
configs['far_plane'] = 20000
configs['encoder_features'] = 32
configs['stereo_baseline'] = 0.54
configs['encoder_ouput_features'] = 64
configs['input_channels'] = 3
configs['out_put_channels'] = 3
configs['num_features'] = 16
configs['occlusion_levels'] = 2
configs['machine_name'] = 'geneva'
configs['use_disc'] = False
configs['max_baseline'] = 4


# Dataset related settings
is_teddy = False
exp = 'spaces_single_view'
print('Experiment: ',exp)
if is_teddy:
    if configs['machine_name'] == 'geneva':
        configs['logging_dir'] = '/netscratch/teddy/monocular_view_synthesis_expts/single_view_carla'
    else:
        configs['logging_dir'] = '/habtegebrialdata/monocular_view_synthesis_expts/single_view_carla'
else:
    # configs['dataset_root'] = '/home5/anwar/data/realestate10k'
    configs['dataset_root'] = '/home5/anwar/data/spaces/spaces_dataset/data'
    configs['logging_dir'] = '/home5/anwar/data/experiments/' + exp
    configs['tensorboard_dir'] = '/home5/anwar/data/experiments/runs/' + exp

os.makedirs(configs['logging_dir'], exist_ok=True)
os.makedirs(configs['tensorboard_dir'], exist_ok=True)
writer = SummaryWriter(configs['tensorboard_dir'])

configs['mode'] = 'train'
configs['num_epochs'] = 10

train_dataset = SpacesLoader(configs)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=configs['batch_size'],
                          shuffle=True,
                          pin_memory=True,
                          num_workers=max(1, configs['batch_size']),
                          drop_last=True
                          )



# train_dataset = RealEstateLoader(configs)
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=configs['batch_size'],
#                           shuffle=True,
#                           pin_memory=True,
#                           num_workers=max(1, configs['batch_size']),
#                           drop_last=True
#                           )

# train_dataset = CarlaLoader(configs, mode='train')
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=configs['batch_size'],
#                           shuffle=True,
#                           pin_memory=True,
#                           num_workers=max(1, configs['batch_size']),
#                           )
# test_dataset = CarlaLoader(configs, mode='test')
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=1,
#                          pin_memory=True,
#                          shuffle=False,
#                          )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
gen_optimizer = torch.optim.Adam(
    monocular_nvs_network.parameters(), lr=gan_opts.lr_gen, betas=(0.9, 0.999))
gen_optimizer.zero_grad()

if configs['use_disc']:
    discriminator = MultiscaleDiscriminator(gan_opts).cuda(0)
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=gan_opts.lr_disc, betas=(0.9, 0.999))
    disc_optimizer.zero_grad()

trainer = Trainer(gan_opts, configs).cuda(0)
if configs['use_disc']:
    trainer.initialise(monocular_nvs_network, discriminator)
else:
    trainer.initialise(monocular_nvs_network, None)

models_dir = os.path.join(configs['logging_dir'], 'models')
os.makedirs(models_dir, exist_ok=True)
steps = 1
error = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)
scheduler = lr_scheduler.LambdaLR(gen_optimizer, lr_lambda= lambda steps: 1.0 - 0.05 * (steps // 500))
for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        steps += 1
        data = {k: v.float().cuda(0) for k, v in data.items()}
        # torch.cuda.synchronize()
        # t_start = time.time()
        gen_losses = trainer(data, mode='generator')
        # torch.cuda.synchronize()
        # print('forward', time.time() - t_start)

        gen_l = sum([v for k, v in gen_losses.items()]).mean()

        gen_print = {k: v.item() for k, v in gen_losses.items()}
        gen_print['Mean'] = gen_l.item()
        print(f'epoch {epoch} iteration {itr}     generator  loss {gen_print}')

        gen_l.backward()
        error += gen_l.item()
        # torch.cuda.synchronize()
        # print('backward', time.time() - t_start)
        utils.clip_grad_norm_(monocular_nvs_network.parameters(), 2.0)
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        scheduler.step()
        # torch.cuda.synchronize()
        # print('update', time.time() - t_start)

        if configs['use_disc']:
            disc_losses = trainer(data, mode='discriminator')
            disc_l = sum([v for k, v in disc_losses.items()]).mean()
            disc_l.backward()
            disc_optimizer.step()
            disc_optimizer.zero_grad()
        novel_view = trainer.fake

        if configs['use_disc']:
            disc_print = {k: v.item() for k, v in disc_losses.items()}
            print(f'epoch {epoch} iteration {itr}  discriminator loss {disc_print}')

        # if(steps % 500 == 0):

        if(steps % 20 == 0):            # writer.add_scalar('Scalar', steps, steps)
            writer.add_scalar('Mean Generator Error', error/300.0, steps)
            error = 0
            novel_view = novel_view.data.cpu()
            target = data['target_img'].data.cpu()
            input_img = data['input_img'].data.cpu()
            torchvision.utils.save_image(novel_view, os.path.join(
                configs['logging_dir'], str(steps) + '_novel.png'))
            torchvision.utils.save_image((target + 1.0)/2.0, os.path.join(
                configs['logging_dir'], str(steps) + '_target.png'))
            torchvision.utils.save_image((input_img + 1.0)/2.0, os.path.join(
                configs['logging_dir'], str(steps) + '_input.png'))
    # if(epoch % 1000 == 0):
    torch.save(monocular_nvs_network.state_dict(), os.path.join(
        models_dir, str(epoch).zfill(4) + 'gen_snapshot.pt'))
    if configs['use_disc']:
        torch.save(discriminator.state_dict(), os.path.join(
            models_dir, str(epoch).zfill(4) + 'disc_snapshot.pt'))
writer.close()
        # here you can do tests every epoch
