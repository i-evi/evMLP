**Note:** This work is **not** related to event cameras.

Code and Pretrained Models for: **[evMLP: An Efficient Event-Driven MLP Architecture for Vision](https://arxiv.org/abs/2507.01927)**

This is a highly experimental implementation of evMLP. Training code, pre-trained models, and evaluation scripts will be updated in the near future.


## Dependencies
Please refer to "requirements.txt". If you don't want to install dependencies according to "requirements.txt", `torch>=2.0.0` is necessary, and it's better to install the latest versions of `einops` and `thop` to support operators and correctly calculate the cost.

## Trainning

You can train on ImageNet-1K by modifying the [references/classification/train.py](https://github.com/pytorch/vision/blob/main/references/classification/train.py) in [torchvision](https://github.com/pytorch/vision).

To train the models in the paper (default configuration in `evmlp.py`), you can use the following settings (using 4 GPUs):

```bash
torchrun --nproc_per_node=4 \
   train.py \
  --auto-augment imagenet \
  --label-smoothing 0.1 \
  --random-erase=0.1 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --epochs 300 \
  --batch-size 256 \
  --opt sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --lr-scheduler cosineannealinglr \
  --lr-min 0.00001 \
  --lr-warmup-method=linear \
  --lr-warmup-epochs=5 \
  --workers 8 \
  --wd 0.00001 \
  --data-path /path/to/dataset
```

## Pre-trained models

Here are the pre-trained models:

[Google Drive](https://drive.google.com/drive/folders/11Ial9MyZJWf4vmd39JKljzv4Ea-Pbr_1?usp=drive_link)

Models trained under the old configuration (deprecated):
* <del>`evmlp_b_224_imagenet1k.pth`: Using the default configuration in `evmlp.py`, trained from scratch on ImageNet-1K.</del>

Available models:

* `evmlp_224_imagenet1k_ep300_73.5.pth`:  Using the default configuration in `evmlp.py`, trained from scratch on ImageNet-1K.

* `evmlp_224_imagenet1k_ep300_distilled_75.4.pth`: Use EfficientNetV2-S(top-1 81.31%@224x224) as the teacher model for knowledge distillation.

## Video processing

Process videos using `eval_video_dir.py`:

```bash
python eval_video_dir.py <weights.pth> <dir_path> <event_threshold>
```

For example, download the model file `evmlp_b_224_imagenet1k.pth`, place the video files in `/path/to/videos`, and use an event threshold of `0.05`:


```bash
python eval_video_dir.py evmlp_b_224_imagenet1k.pth /path/to/videos 0.05
```

`eval_video_dir.py` uses `opencv_python` to load video files. The default filter list only supports video files with extensions `.avi` and `.mp4`. If necessary, you can edit the following code:

```python
L31@eval_video_dir.py: video_extensions = {'.avi', '.mp4'}
```

## FAQs

**Q**: Can evMLP be used for other computer vision tasks besides image classification?

**A**: Certainly. The feature maps reconstructed by `evMLP` through the rearrange operation can maintain the adjacency relationship between neuron patches relative to the input image, making it directly applicable to tasks such as object detection and segmentation. If I have time, I will update some examples of applying `evMLP` to other tasks.

**Q**: Why has the number of MACs decreased, but the execution time increased instead?

**A**: This repository only provides experimental Python code. If you understand that:

Code 1:

```python
a = numpy.random.rand(N)
sum = 0.
for i in a:
  sum += i

```

Code 2:

```python
a = numpy.random.rand(N)
sum = a.sum()
```
Even though both codes sum the array `a`, the execution time of `Code 2` might be significantly shorter than `Code 1`. For practical applications, the code can be implemented in C/C++. Alternatively, using FPGA for implementation is also a great option.

