import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import GANLoss
from .losses import SynthesisLoss
from .losses import MultiscaleDiscriminator

class Trainer(nn.Module):
    def __init__(self, opts):
        super(Trainer, self).__init__()
        self.configs = opts
        self.synthesis_loss = SynthesisLoss(opts)
        self.get_gan_loss = GANLoss(opts.gan_mode)

    def initialise(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
    def forward(self, input_data, mode='generator'):
        if mode=='generator':
            novel_view, alpha = self.generate_fake(input_data)
            # gan_losses = self.compute_generator_loss(novel_view, input_data['target_img'])
            self.fake = self.to_image(novel_view.data)
            self.real = self.to_image(input_data['target_img'].data)
            synthesis_losses = self.synthesis_loss(novel_view, input_data['target_img'])
            # gan_losses.update(synthesis_losses)
            return synthesis_losses, novel_view, alpha
            # return gan_losses, novel_view, alpha
        elif mode=='discriminator':
            gan_losses = self.compute_discriminator_loss(input_data)
            return gan_losses
        elif mode=='inference':
            with torch.no_grad:
                if not ('target_img' in input_data.keys()):
                    input_data['target_img'] = None
                novel_view, alpha = self.generate_fake(input_data)
            return novel_view
        else:
            raise KeyError('Mode should be in [generator, discriminator, inference]')

    def generate_fake(self, input_data):
        novel_view, alpha = self.generator(input_data['input_img'], input_data['k_mats'], input_data['r_mats'], input_data['t_vecs'])
        return novel_view, alpha

    def compute_discriminator_loss(self, data):
        target_img = data['target_img']
        input_img = data['input_img']
        D_losses = {}
        with torch.no_grad():
            if not ('target_img' in data.keys()):
                data['target_img'] = None
            fake_image, alpha = self.generate_fake(data)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(fake_image, target_img)
        D_losses['D_Fake'] = self.get_gan_loss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.get_gan_loss(pred_real, True, for_discriminator=True)
        return D_losses

    # def compute_generator_loss(self, fake_image, target_img):
        device_ = fake_image.device
        gen_losses = {}
        # print('compute_generator_loss:', fake_image.shape, target_img.shape)
        pred_fake, pred_real = self.discriminate(fake_image, target_img)
        gen_losses['GAN'] = sum(self.get_gan_loss(pred_fake, True, for_discriminator=False))
        if not self.configs.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(device_)
            for i in range(num_D):
                # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs): # for each layer output
                    unweighted_loss = F.l1_loss(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.configs.lambda_feat / num_D
            gen_losses['GAN_Feat'] = sum(GAN_Feat_loss)
        return gen_losses

    def discriminate(self, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.discriminator(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def to_image(self, tensor):
        img = (tensor+1.0)/2.0
        img = img.clamp(min=0.0, max=1.0)
        return img
