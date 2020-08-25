import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--d_step_per_g', type=int, default=1, help='num of d updates for each g update')
arg_parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
arg_parser.add_argument('--spade_k_size', type=int,default=3)
arg_parser.add_argument('--num_D', type=int, default=3)
arg_parser.add_argument('--output_nc', type=int, default=3)
arg_parser.add_argument('--n_layers_D', type=int, default=4)
arg_parser.add_argument('--contain_dontcare_label', default=False, type=bool)
arg_parser.add_argument('--no_instance', default=True, type=bool)

arg_parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')
# arg_parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='instance normalization or batch normalization')
arg_parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
arg_parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

# Generator settings
arg_parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
arg_parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
# Discriminator setting
arg_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
arg_parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
arg_parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
arg_parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
arg_parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
arg_parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
arg_parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
# arg_parser.set_defaults(norm_G='')
arg_parser.add_argument('--num_upsampling_layers',
                    choices=('normal', 'more', 'most'), default='normal',
                    help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
arg_parser.add_argument('--num_out_channels', default=3, type=int)
arg_parser.add_argument('--losses', type=str, nargs='+', default=['1.0_l1','10.0_content'])
arg_parser.add_argument('--lr_gen', default=0.0004, type=float)
arg_parser.add_argument('--lr_disc', default=0.0001, type=float)
