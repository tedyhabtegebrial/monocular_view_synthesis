## MPI

Different modules for handling the scene in the MPI representation

#### Alpha Composition

- Implements the `over` operation back to front


#### Compute Homography

- Computes the Homography matrix `H` from source view to target view

#### Apply Homography

- Applies the Homography matrix `H` on a scene in MPI representation to transform it from the source view where it is estimated into the target view


#### Compute Blending Weights

- Computes the blending weights as a function of the estimated alphas 
