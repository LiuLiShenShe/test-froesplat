# Training

This section provides the instructions and details of dataset for training the AMB3R model.

### Training Data

 AMB3R is trained on a mixture of 11 datasets. For each training epoch. we sample a total of **2,000 samples** , where each sample consists of **5 to 16 frames**. The sampling distribution across different scene types is balanced as follows: 

* **🏠 Indoor (700 samples):** ScanNet++, ScanNet, Aria, Hypersim
* **📦 Object-Centric (300 samples):** WildRGBD, OmniObject3D
* **🚗 Autonomous Driving (600 samples):** Waymo, Virtual KITTI 2
* **🌳 Outdoor (400 samples):** MapFree, GTASfM, MVS-Synth


We provide an example ScanNet dataset class in `data/scannet.py`, which is already integrated into the training script. To add more datasets, please add your custom dataset classes to the `data/` directory and update `amb3r/training.py` accordingly.



### Training Instruction

#### AMB3R Training

1. Download the pretrained VGGT weights and place it under `./checkpoints/VGGT.pt`

2. Launch the training script using `torchrun`. Please replace `$num_gpus` and `$batch_size` with the actual number of GPUs available on your node and choose proper batch size for your GPU type:

```sh
torchrun --nproc_per_node $num_gpus train.py --batch_size $batch_size
```



**NOTE:** In our currently released pre-trained checkpoints, the model was trained using a KNN interpolation implementation that contains a slight misalignment between the voxel coordinate system and the point coordinate system. However, since the offset information between these two coordinate systems is contained in the input feature to the backend transformer, the network learns to compensate for this small offset internally. 

If you plan to train the model for other tasks from scratch, you can use the updated interpolation logic (which removes the misalignment) by appending the --interp_v2 flag to your training command:

```sh
torchrun --nproc_per_node $num_gpus train.py --batch_size $batch_size --interp_v2
```



#### Metric-scale Estimator Training

Once you finish the training of amb3r, you can launch the following script for training metric-scale estimator:

```sh
torchrun --nproc_per_node $num_gpus train_metric.py --pretrained $path_to_your_best_ckpt
```



#### 

