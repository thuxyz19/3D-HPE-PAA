# Boosting Monocular 3D Human Pose Estimation with Part Aware Attention

## Environment settings
The codebase is tested under the following environment settings:
- cuda: 11.0
- python: 3.8.10
- pytorch: 1.7.1
- torchvision: 0.8.2
- scikit-image: 0.18.2
- einops: 0.3.0
- timm: 0.4.9
- pyyaml: 6.0
- easydict: 1.9
- opencv-python: 4.5.2.54


## Prepare the dataset
To perform the evaluation on the Human3.6M dataset, you should: 

1. Download data.zip from https://cloud.tsinghua.edu.cn/f/b102a975ff8d4ae1a4c1/?dl=1. 
2. Extract the file.
3. Put the extracted files into ./data/ directory.

After doing so, the file structure should be as follows:

    ./data
        data_2d_h36m_cpn_ft_h36m_dbb.npz
        data_2d_h36m_gt.npz
        data_3d_h36m.npz

## Download the checkpoints
The trained checkpoints can be downloaded from https://cloud.tsinghua.edu.cn/d/fae76890154a45a99b31/. After downloaded, the checkpoints should be put into the ./checkpoint/ directory and the file structure of ./checkpoint/ should be as follows.

    ./checkpoint
        cpn_f81.bin
        cpn_f243.bin
        gt_f81.bin
        gt_f243.bin

## Evaluate
To conduct evaluation using the CPN inputs, you can run the following commands:

```shell
CUDA_VISIBLE_DEVICES=0 python eval.py -c ./exp/exp_cpn_f81.bin # using 81 frames as input
```
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py -c ./exp/exp_cpn_f243.bin # using 243 frames as input
```

Similarly, to evaluate using the GT inputs, you can run the following commands:

```shell
CUDA_VISIBLE_DEVICES=0 python eval.py -c ./exp/exp_gt_f81.bin # using 81 frames as input
```
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py -c ./exp/exp_gt_f243.bin # using 243 frames as input
```


