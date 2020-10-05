## Monocular View Synthesis with Stereo Magnification


#### Dependencies
PyTorch:
      torch>=1.4.0

#### Usage
#### Simple test

```
# to run the model with dummy data
$ cd monocular
$ python -m src.monocular_stereo_magnification

```


### How to use it with your code

Create a config dictionary and a data loader that conforms the conventions followed below

#### Set up a configuration dictionary

```python
configs = {}
configs['width'] = 128 # 512    # image width
configs['height'] = 128 # 256   # image height
configs['batch_size'] = 1
configs['num_planes'] = 64 # number of planes to represent the scene geometry
configs['near_plane'] = 5  # the closest plane in meters
configs['far_plane'] = 10000 # the farthest plane, in meters
# This controls the size of the Encoder Decoder backend
configs['encoder_features'] = 32
# This controls the size of the prediction head
configs['encoder_ouput_features'] = 64
# here we specify the number of channels in the input and output images
configs['input_channels'] = 3
configs['out_put_channels'] = 3

## Dataset related settings
configs['dataset_root'] = '/data/Datasets/KittiOdometry/dataset'
# mode t['train', 'test']
configs['mode'] = 'train'
# maximum frame-distance between input and target frames
configs['max_baseline'] = 5

```
#### Create an instance of StereoMagnification class with the configs dictionary

```python
monocular_nvs_network = StereoMagnification(configs).eval()
```
#### Load data in the folowing fashion
  * ```k_mats```: is the camera intrinsics (right now I am assuming the same matrix for input and target cameras)
  * ```r_mats```: rotation matrix from source camera to the target camera
  * ```t_vecs```: translation vector, it is the source camera center as seen from the target
```python
input_img = torch.rand(1, 3, 128, 128)
k_mats = torch.rand(1,3,3)
r_mats = torch.rand(1,3,3)
t_vecs = torch.rand(1,3,1)
```
```python
novel_view = monocular_nvs_network(input_img, k_mats, r_mats, t_vecs)
print(f'Novel VIew Shape== {novel_view.shape}')
# you should get [B, 3, H, W]
```

#### Training on different architectures
There are two implementations for the network and you can choose by changing the `self.mpi_net` in `monocular_stereo_magnification.py`:
* SingleViewNetwork: imitates the approach of the SingleViewMPI paper and in order to use it:
  * Set `self.mpi_net` to `SingleViewNetwork` and use the `forward` function implemented at line `62`
* SingleViewNetwork_DFKI: implements our approach and in order to use it:
 * set `self.mpi_net` to `SingleViewNetwork_DFKI` and use the `forward` function implemented in line `47`


#### Limitations

1. Supports planes which are fronto-parallel to the source camera
2. The source and target camera intrinsics are assumed to be the same
