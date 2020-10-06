import os
import sys
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
from monocular.src import SingleViewNetwork_DFKI
from monocular.src import CarlaLoader
gan_opts = arg_parser.parse_args()

configs = {}
configs['width'] = 256
configs['height'] = 256
configs['batch_size'] = 1
configs['num_planes'] = 32
configs['near_plane'] = 5
configs['far_plane'] = 20000
configs['encoder_features'] = 32
configs['stereo_baseline'] = 0.54
configs['encoder_ouput_features'] = 64
configs['input_channels'] = 3
configs['out_put_channels'] = 3
configs['num_features'] = 4
configs['occlusion_levels'] = 3
configs['machine_name'] = 'geneva'
configs['use_disc'] = False


# Dataset related settings
is_teddy = True

if is_teddy:
    if configs['machine_name'] == 'geneva':
        configs['logging_dir'] = '/netscratch/teddy/monocular_view_synthesis_expts/single_view_carla'
    else:
        configs['logging_dir'] = '/habtegebrialdata/monocular_view_synthesis_expts/single_view_carla'
else:
    configs['logging_dir'] = '/home/anwar/data/experiments/exp5'

os.makedirs(configs['logging_dir'], exist_ok=True)

configs['mode'] = 'train'
configs['num_epochs'] = 30

train_dataset = CarlaLoader(configs, mode='train')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=configs['batch_size'],
                          shuffle=True,
                          num_workers=max(1, configs['batch_size'] // 2),
                          )
test_dataset = CarlaLoader(configs, mode='test')
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         )

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
steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)

for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        data = {k: v.float().cuda(0) for k, v in data.items()}
        gen_losses = trainer(data, mode='generator')
        gen_l = sum([v for k, v in gen_losses.items()]).mean()
        gen_l.backward()
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        if configs['use_disc']:
            disc_losses = trainer(data, mode='discriminator')
            disc_l = sum([v for k, v in disc_losses.items()]).mean()
            disc_l.backward()
            disc_optimizer.step()
            disc_optimizer.zero_grad()
        novel_view = (trainer.fake + 1.0) / 2.0
        gen_print = {k: v.item() for k, v in gen_losses.items()}
        print(f'epoch {epoch} iteration {itr}     generator  loss {gen_print}')
        if configs['use_disc']:
            disc_print = {k: v.item() for k, v in disc_losses.items()}
            print(
                f'epoch {epoch} iteration {itr}  discriminator loss {disc_print}')
        if(steps % 200 == 0):
            novel_view = novel_view.data[:, [2, 1, 0], :, :].cpu()
            target = (data['target_img'].data[:, [
                      2, 1, 0], :, :].cpu() + 1) / 2.0
            input_img = (data['input_img'].data[:, [
                         2, 1, 0], :, :].cpu() + 1) / 2.0
            torchvision.utils.save_image(novel_view, os.path.join(
                configs['logging_dir'], str(steps) + '_novel.png'))
            torchvision.utils.save_image(target, os.path.join(
                configs['logging_dir'], str(steps) + '_target.png'))
            torchvision.utils.save_image(input_img, os.path.join(
                configs['logging_dir'], str(steps) + '_input.png'))
        steps += 1

    torch.save(monocular_nvs_network.state_dict(), os.path.join(
        models_dir, str(epoch).zfill(4) + 'gen_snapshot.pt'))
    if configs['use_disc']:
        torch.save(discriminator.state_dict(), os.path.join(
            models_dir, str(epoch).zfill(4) + 'disc_snapshot.pt'))
        # here you can do tests every epoch
