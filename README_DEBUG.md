## 20250325

1. sam2 反复调用，同一进程显存迟早会爆炸

解决办法：使用 shell 脚本反复运行`inference.py`文件


## legacy

TODO:

- 使用astype会导致阶段，测试一下哪个方法准确度更高

1. 增加whole图像的拆封和预处理；

2. 增加预处理脚本将图像从TIF -> RGB

3. CUDA OUT OF MEM

4. 增加前端处理脚本

5. Mask提取后计算重复率


---
## 20250123

1. 使用photoshop截取相同pix的WFA和NeuN，使用PNG进行识别



## 20250122


1. 已经将Hole个数计算算法部署完成（SAM2），但存在计算bug：在计算个数的时候，会默认给全局一个mask，需要将全局mask减去

2. 计算重合度算法：IoU， DICE



## 20250121

1. adjust hyper-parameters in DL model

```

```

## 20250120

1. Test SAM2 for segmentation

## 20250107

1. SAM 模型部署，测试小批量拆分图像数据

2. 成功部署模型SAM2

## 20250106

1. 出现bug`Cuda out of memory`，内存吃紧，可能是图像过大，或者模型太大；选择拆模型或者拆图像；

2. 采用DDPM技术提高内存使用率


## 20250105

1. 图像缩小，识别小区域

## 20250104

1. 数据整理：归一化，梯度值分析

## 20250103

test git

add README.md
