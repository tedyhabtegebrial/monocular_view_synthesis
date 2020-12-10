## Data sets

#### Common returned Dictionary

```
'input_img': image in the source view
'target_img': image in the target view
'diff_kmats': flag whether the intrinsics of the target camera and the source camera are the same
'k_mats': source camera intrinsics
'k_mats_target': target camera intrinsics if 'diff_kmats' is true
'r_mats', 't_vecs': transformation matrices from source view to target view
```

#### Datasets

- Carla: https://carla.org/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- RealEstate10k: https://google.github.io/realestate10k/
- Spaces: https://github.com/augmentedperception/spaces_dataset
