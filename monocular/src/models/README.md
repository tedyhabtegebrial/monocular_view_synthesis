## Models

#### Single View Model

Based on the paper `Single-view view synthesis with multiplane images` : https://single-view-mpi.github.io/

- Direct implementation of the same model as described in the paper
- Estimates alpha values of the MPI as well as background RGB

#### Single View - DFKI

- Our own implementation of the Single View model
- Estimate alpha values for each plane as well as a number of `occlusion_levels` for each plane which as an estimate of different levels of visibility/occlusion inside the scene

##### Background Network

- Helper network for our implementation
- Transfers the input RGB image to a higher dimension in order to encode more information about the scene before projecting it on the MPI
- Transfer an image from a higher dimension back to RGB
- Mode of operation depends on the flag `reduce`
- Higher dimension depends on the configuration value `num_features`
