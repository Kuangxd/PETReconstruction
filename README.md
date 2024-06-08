# PET Image Reconstruction Using Weighted Nuclear Norm Maximization and Deep Learning Prior

#### Subject to intellectual property rights, the modules that can accelerate the forward and backward propagation of PET images cannot be open-sourced. We provide an alternative simulation framework for researchers to conduct experiments. The image matrix size is set to 128×128×4

## Getting Started
- Modify the model save path (--checkpoints_dir) in the `01-TICCGAN/options/base_options.py` file according to the user's running environment.
- Modify the system path in the `00-ncamar-release/dl_function.py` file according to the user's running environment.

### Testing our method
- Set w_svd to 0.03 in `00-pet.py`, and then run
```bash
python 00-pet.py
```

### Testing iterative CNN
- Set w_svd to 0 in 00-pet.py, and then run
```bash
python 00-pet.py
```
