__all__ = ['StereoMagnification', 'KittiLoader',
           'CarlaLoader', 'Trainer', 'SingleViewNetwork_DFKI']
from .monocular_stereo_magnification import StereoMagnification
from .models import SingleViewNetwork_DFKI
from .datasets import CarlaLoader, SpacesLoader
from .datasets import KittiLoader, RealEstateLoader
from .trainer import Trainer
from .losses import MultiscaleDiscriminator
