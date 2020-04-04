## Monocular View Synthesis with Stereo Magnification


#### Dependencies
PyTorch:
      torch>=1.4.0


#### Usage

#### Set up a configuration dictionary

```python
configs = {}
configs['width'] = 128 # 512    # image width
configs['height'] = 128 # 256   # image height
configs['batch_size'] = 1
configs['num_planes'] = 64 # number of planes to represent the scene geometry
configs['near_plane'] = 5  # the closest plane in meters
configs['far_plane'] = 10000 # the farthest plane, in meters
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
input_img = torch.rand(1, 3, 256, 256)
k_mats = torch.rand(1,3,3)
r_mats = torch.rand(1,3,3)
t_vecs = torch.rand(1,3,1)
```
```python
novel_view = monocular_nvs_network(input_img, k_mats, r_mats, t_vecs)
print(f'Novel VIew Shape== {novel_view.shape}')
# you should get [B, 3, H, W]
```
