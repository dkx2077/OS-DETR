
# Orthogonal Channel Shuffle: A Novel Approach to Accurate Detection of Brain Tumors



## Environment Setup

<!-- <details> -->
<!-- <summary>Click to expand</summary> -->

1. Uninstall the ultralytics library from your environment:
   ```bash
   pip uninstall ultralytics
   ```
   Note: If you're also using YOLOv8, it's recommended to create a separate virtual environment using Anaconda for this project to avoid conflicts.

2. Run the uninstall command again to ensure complete removal. If you see the message "WARNING: Skipping ultralytics as it is not installed.", it means the library has been successfully uninstalled.

3. (Optional) If you need to use the official CLI running method, install the ultralytics library by running:
   ```bash
   python setup.py develop
   ```
   This step can be skipped if you don't need the official CLI running method.

4. Install additional required packages:
   ```bash
   pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv
   ```

   The following packages are necessary for using DyHead:
   ```bash
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"
   ```
   Note: If these packages fail to install, DyHead cannot be used normally!

5. If you encounter any missing package errors during runtime, please install them as needed.

<!-- </details> -->

## Usage
1. Test OS Moudle
```python
python OS_Moudle_test.py
```

2. Start Val
We provide the trained completed model used in the paper under `./runs/train/OS`

```python
python val.py
```

3. Start Train
```python
python train.py
```

</details>




