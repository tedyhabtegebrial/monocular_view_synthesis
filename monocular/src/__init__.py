__all__ = ['StereoMagnification', 'KittiLoader', 'Trainer']
from .monocular_stereo_magnification import StereoMagnification
from .datasets import KittiLoader, RealEstateLoader
from .trainer import Trainer
from .losses import MultiscaleDiscriminator
