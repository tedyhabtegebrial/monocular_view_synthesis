<<<<<<< Updated upstream
import os, sys, time
=======
import os
import sys
>>>>>>> Stashed changes
import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils
import torch.nn.utils
from tensorboardX import SummaryWriter
from .gan_opts import arg_parser
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from monocular.src import StereoMagnification
from monocular.src import KittiLoader, RealEstateLoader
from monocular.src import Trainer
from monocular.src import MultiscaleDiscriminator
gan_opts = arg_parser.parse_args()

configs = {}
configs['width'] = 384
configs['height'] = 256
configs['batch_size'] = 2
configs['num_planes'] = 32
configs['near_plane'] = 5
configs['far_plane'] = 20000
configs['encoder_features'] = 32
configs['encoder_ouput_features'] = 64
configs['input_channels'] = 3
configs['out_put_channels'] = 3
configs['num_features'] = 16
configs['occlusion_levels'] = 3

<<<<<<< Updated upstream
## Dataset related settings
is_teddy = False
=======
# Dataset related settings
is_teddy = True
>>>>>>> Stashed changes
if is_teddy:
    # configs['dataset_root'] = '/home/anwar/data/KITTI_Odometry/dataset'
    configs['dataset_root'] = '/data/teddy/KITTI_Odometry/dataset'
    configs['logging_dir'] = '/habtegebrialdata/monocular_nvs/experiment_logs/exp_1_with_bn_new_alpha_comp'
else:
    configs['dataset_root'] = '/home5/anwar/data/realestate10k'
    configs['logging_dir'] = '/home5/anwar/data/experiments/exp_SV_GAN_LOSS_1'

configs['mode'] = 'train'
configs['max_baseline'] = 26
configs['num_epochs'] = 10

tb_path = os.path.join(configs['logging_dir'], 'runs')
os.makedirs(tb_path, exist_ok=True)
writer = SummaryWriter(tb_path)

#train_dataset = KittiLoader(configs)
train_dataset = RealEstateLoader(configs)
train_loader = DataLoader(dataset=train_dataset,
<<<<<<< Updated upstream
                         batch_size=configs['batch_size'],
                         shuffle=True,
                         num_workers=max(1, configs['batch_size']//2),
                         )
#test_dataset = KittiLoader({**configs, 'mode':'test'})
#test_loader = DataLoader(dataset=test_dataset,
#                         batch_size=1,
#                         shuffle=False,
#                         )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
gen_optimizer = torch.optim.Adam(monocular_nvs_network.parameters(), lr=gan_opts.lr_gen, betas=(0.9, 0.999))
=======
                          batch_size=configs['batch_size'],
                          shuffle=True,
                          num_workers=max(1, configs['batch_size'] // 2),
                          )
test_dataset = KittiLoader({**configs, 'mode': 'test'})
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         )

monocular_nvs_network = StereoMagnification(configs).float().cuda(0)
gen_optimizer = torch.optim.Adam(
    monocular_nvs_network.parameters(), lr=gan_opts['lr_gen'], betas=(0.9, 0.999))
>>>>>>> Stashed changes
gen_optimizer.zero_grad()


discriminator = MultiscaleDiscriminator(gan_opts).cuda(0)
<<<<<<< Updated upstream
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan_opts.lr_disc, betas=(0.9, 0.999))
=======
disc_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=gan_opts['lr_disc'], betas=(0.9, 0.999))
>>>>>>> Stashed changes
disc_optimizer.zero_grad()

trainer = Trainer(gan_opts).cuda(0)
trainer.initialise(monocular_nvs_network, discriminator)


models_dir = os.path.join(configs['logging_dir'], 'models')
os.makedirs(models_dir, exist_ok=True)
steps = 0
print('Logging info: ', configs['logging_dir'])

torch.autograd.set_detect_anomaly(True)

a = 0
b = 1
min_val = -1
max_val = 1

for epoch in range(configs['num_epochs']):
    print(f'Epoch number = {epoch}')
    for itr, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
<<<<<<< Updated upstream
        data = {k:v.float().cuda(0) for k,v in data.items()}
#        torch.cuda.synchronize()
#        start = time.time()
        gen_losses = trainer(data, mode='generator')[0]
#        torch.cuda.synchronize()
#        print('time 1', time.time() - start)
        gen_l = sum([v for k,v in gen_losses.items()]).mean()
        # gen_l = gen_losses['Total Loss']  + gen_losses['GAN'] * gan_opts.lamda_gan
        print('gen_l', gen_l.item())
=======
        data = {k: v.float().cuda(0) for k, v in data.items()}
        gen_losses = trainer(data, mode='generator')
        gen_l = sum([v for k, v in gen_losses.items()]).mean()
>>>>>>> Stashed changes
        gen_l.backward()
        gen_optimizer.step()
        gen_optimizer.zero_grad()
#        torch.cuda.synchronize()
#        start = time.time()
        disc_losses = trainer(data, mode='discriminator')
<<<<<<< Updated upstream
#        torch.cuda.synchronize()
       # print('time 2', time.time() - start)
        disc_l = sum([v for k,v in disc_losses.items()]).mean()
        disc_l.backward()
        disc_optimizer.step()
        disc_optimizer.zero_grad()
        novel_view = trainer.fake
        gen_print = {k:v.item() for k,v in gen_losses.items()}
        disc_print = {k:v.item() for k,v in disc_losses.items()}
        print(f'epoch {epoch} iteration {itr}     generator  loss {gen_print}')
        print(f'epoch {epoch} iteration {itr}  discriminator loss {disc_print}')
        if(steps % 300 == 0):
            novel_view = novel_view.data.cpu()
            target = data['target_img'].data.cpu()
            input_img = data['input_img'].data.cpu()
            torchvision.utils.save_image(novel_view, os.path.join(configs['logging_dir'], str(steps) +'_novel.png'))
            # writer.add_image('Novel View', novel_view[0], steps)
            #writer.add_scalar('Scalar', steps, steps)
            torchvision.utils.save_image(target, os.path.join(configs['logging_dir'], str(steps) +'_target.png'))
            # writer.add_image('Target View', target[0], steps)
            torchvision.utils.save_image(input_img, os.path.join(configs['logging_dir'], str(steps) +'_input.png'))
            # writer.add_image('Input View', input_img[0], steps)
=======
        disc_l = sum([v for k, v in disc_losses.items()]).mean()
        disc_l.backward()
        disc_optimizer.step()
        disc_optimizer.zero_grad()
        novel_view = (trainer.fake + 1.0) / 2.0
        gen_print = {k: v.item() for k, v in gen_losses.items()}
        disc_print = {k: v.item() for k, v in disc_losses.items()}
        print(f'epoch {epoch} iteration {itr}     generator  loss {gen_print}')
        print(f'epoch {epoch} iteration {itr}  discriminator loss {disc_print}')
        if(steps % 200 == 0):
            novel_view = novel_view.data[:, [2, 1, 0], :, :].cpu()
            target = data['target_img'].data[:, [2, 1, 0], :, :].cpu()
            input_img = data['input_img'].data[:, [2, 1, 0], :, :].cpu()
            torchvision.utils.save_image(novel_view, os.path.join(
                configs['logging_dir'], str(steps) + '_novel.png'))
            torchvision.utils.save_image(target, os.path.join(
                configs['logging_dir'], str(steps) + '_target.png'))
            torchvision.utils.save_image(input_img, os.path.join(
                configs['logging_dir'], str(steps) + '_input.png'))
>>>>>>> Stashed changes
        steps += 1
        # exit()

<<<<<<< Updated upstream
    #writer.export_scalars_to_json(os.path.join(tb_path,'all_scalars.json')
    writer.close()
    torch.save(monocular_nvs_network.state_dict(), os.path.join(models_dir, str(epoch).zfill(4)+'gen_snapshot.pt'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, str(epoch).zfill(4)+'disc_snapshot.pt'))
    #### here you can do tests every epoch
=======
    torch.save(monocular_nvs_network.state_dict(), os.path.join(
        models_dir, str(epoch).zfill(4) + 'gen_snapshot.pt'))
    torch.save(discriminator.state_dict(), os.path.join(
        models_dir, str(epoch).zfill(4) + 'disc_snapshot.pt'))
    # here you can do tests every epoch
>>>>>>> Stashed changes
