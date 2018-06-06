---
layout: post
title: Build up the hardware envirnment for TensorFlow Workstation
category: envirnment
tags: [TensorFlow,GPU, envirenment]
---
### 配置
主板：ASUS X99 -E WS，特点：单CPU,
支持8*16 PCIE通道

CPU： i7 6850k, 40 lanes

内存：16g*2

SSD：256g Sumsung

HDD：2T

GPU： MSI GTX 1080 Ti 一个
##### Reference:
1. [A Full Hardware Guide to Deep Learning](http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/)
1. [深度学习（TensorFlow）环境搭建：（一）硬件选购和主机组装](https://www.cnblogs.com/xuliangxing/p/7543977.html)

### 安装系统攻略
系统预装了win10，占用了SSD 128G， HDD 1T，需要加装ubuntu 16.04。虽然win10系统可以用基础的显卡驱动来通过1080Ti本身达到基本显示的目的，但，如果想在win10上发挥1080Ti的全部威力，依然需要另外安装显卡驱动；第二，只要出了win10系统，显卡就完全不能使用，包括安装ubuntu系统的过程。因此，在安装ubuntu系统地过程中，需要临时借助一个低配显卡（我们使用的是不需要插电，也不需要预装驱动即可运行的GTX 650）来连接主机与显示器，点亮系统。

点亮系统的过程中，请拔下1080Ti，防止系统识别错误。
#### 准备：
unbuntu 16.04系统U盘；
#### 开始：
开机，在显示华硕主板界面的时候按F2进入BOIS模式，切换到通过外置U盘再开机。（请按照引导优先使用键盘上的F1-12进行操作，更不容易出错）
成功后，系统会重启，然后进入U盘中的UbuntuBoot引导界面，选择Install Ubuntu（instead of Try Ubuntu）
选择语言后，continue；然后会有一个回退（go back）的步骤，此时可以选择系统默认安装，还是自定义（something else）。请选择something else，第一次选择默认安装后，系统图形界面除了guest模式，不能成功登陆.

>趁机熟悉了linux的一系列命令行操作：
* sudo apt-get 软件名，可以安装软件
* sudo service lightdm restart/stop
* ctrl+R 可以搜索历史命令行。
* tail -f /var/log/auth.log 可以查看日志，随时观察系统报错。
* ctrl alt F1-6, 可以切换六个不同的terminal界面。 ctrl alt F7 是图形界面


分区时参考了[ubuntu16.04分区设置](https://blog.csdn.net/zhangxiangweide/article/details/74779652)的法1，
在SSD上分出:

* ```/``` 主分区120g（装系统默认配置文件，如pool文件，etc文件等等）
* ```/swap``` 16g（参考内存一共32g）
* ```/boot``` 1.1g（引导盘）
在HDD上:
* 只分了一个盘，分出```/home``` 931.5g。

注意：最下面的引导盘选择，请选择有256g大小的整个SSD盘，而不要仅仅选择/boot 引导分区，装好后，会进入 rescue grubjiemian，并且网上寻找引导盘的set root + set prefix并不能拯救你😂
（附最终没有帮上忙的切换引导盘代码:
```
grub rescue>ls
grub rescue>set root = (hd0,msdos5)
grub rescue>set prefix = (hd0,msdos5)/boot/grub
grub rescue>insmod normal
grub rescue>normal```

（It turns out to be useless）

### GTX1080 Ti 驱动安装攻略

#### 零
安装完系统之后，关闭主机，把1080 Ti插到主板上（记得给尊贵的1080 Ti插电），并且把显示器的连接线从低配版显卡换到 1080 Ti上，__！但是！ 不要把低配显卡拔掉，否则系统仍然无法正确显示。__（驱动还没装成，同志仍需努力！
##### 可跳过的步骤:
（我们是遵循了的，不确定如果不遵循会不会报错，有攻略称可有可无。）

在启动主板时，按F2进入BOIS模式。进入高级选项(F7),找到BOOT模块，找到secure boot，将UEFI windows 关闭成other OS（此处描述仅限华硕X99主板，其他主板的开关secure boot的位置可能略有不同）
然后 press F10 save and exit, the system will reboot.

#### 一、安装ppa库
```sudo add-apt-repository ppa:graphics-drivers/ppa > ppa.tmp```
输入这一行命令之后，系统会告诉你当下比较主流的Nvidia驱动版本。由于打印过长，
使用```> ppa.tmp```把output 导出成文件查看。
目前的稳定版是390，因此我们一会下载nvidia-390。

#### 二、关闭图形化界面环境
```sudo service lightdm stop```
#### 三、 下载
```sudo apt-get update && sudo apt-get install nvidia-390```
#### 四、验证
```nvidia-smi```




##### Reference
1. [深度学习（TensorFlow）环境搭建：（二）Ubuntu16.04+1080Ti显卡驱动
](https://www.cnblogs.com/xuliangxing/p/7569946.html)
1. [Ubuntu16.04 + 1080Ti深度学习环境配置教程](https://www.jianshu.com/p/5b708817f5d8)

### 配置系统环境
##### Reference
1. [深度学习（TensorFlow）环境搭建：（三）Ubuntu16.04+CUDA8.0+cuDNN7+Anaconda4.4+Python3.6+TensorFlow1.3](https://www.cnblogs.com/xuliangxing/p/7575586.html)
