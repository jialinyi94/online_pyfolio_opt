# online_pyfolio_opt
Online porfolio optimization algorithms implemented in Python with Jax

## Pin CUDA version

check [CUDA support-matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#gpu-cuda-toolkit-and-cuda-driver-requirements) for compatibilities.

- cudatoolkit==12.2.0

install from: [for WSL-ubuntu (x86_64)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2
```

check the exact include and lib folders that `/user/local/cuda` is linking


- cudnn==9.4.0

install from [for WSL-ubuntu (22.04 LTS)](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/)

unzip the file and copy the header files in `include` and `lib` to the include and lib folders.
