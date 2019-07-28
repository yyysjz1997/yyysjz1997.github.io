---
layout: post
title:  安装Win10+Ubantu18.04双系统
date:   2019-07-26 15:30:00 +0800
categories: 技术交流
tag: 软件领域
---

* content
{:toc}


准研究生的项目需要设计一个软件界面，之前用过QT和MFC，为了UI方便，这次就直接用QT~

这次得在新的电脑上安装QT，为了防止出现什么意外直接选择之前用的版本（5.11.0），当然了我还是使用之前的编译器Desktop Qt 5.11.0 MinGW 32bit(因为这个不用VS)

结果安装好后出现了如下的报错：

![\styles\images\2019-07-27-install-QT5/1.jpg]({{ '\styles\images\2019-07-27-install-QT5/1.jpg' | prepend: site.baseurl  }}) 

本来以为是新电脑没有装一些驱动，网上一查是因为我开的QT工程的路径下存在中文，改成了英文路径后没有问题喽。

开干！！！