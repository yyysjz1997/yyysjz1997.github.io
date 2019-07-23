---
layout: post
title:  配置深度学习环境
date:   2019-07-23 18:02:00 +0800
categories: 技术交流
tag: 软件领域
---

* content
{:toc}


主机装好，下面就要开始装深度学习环境了，我还是直接用在windows下装吧。。。

首先是用到的东西（显卡是nvidia的RTX 2070 Super）：**anaconda3、python、cuda、cudnn、tensorflow-gpu、pycharm**

比较坑的是，所有的一切需要版本好的匹配。

我自己的软件版本是这样的（配置的时间是2019年7月22号）

- 系统：win10专业版

- anaconda3：2019.03版

- python：3.7.3

- cuda：10.0

- cudnn：7.4.1

- tensorflow-gpu：1.13.1

- pycharm：2019.1.3版

整个的流程就是anaconda(顺便把python、jupyter、spyder)安装好->cuda->cudnn->tensorflow-gpu->pycharm等

所有有用的博客我是参照这些网站的。**[网址一](https://www.jianshu.com/p/9f89633bad57)** **[网址二](https://www.cnblogs.com/guoyaohua/p/9265268.html#%E7%AC%AC%E5%9B%9B%E6%AD%A5%E6%B5%8B%E8%AF%95)** **[网址三](https://www.cppentry.com/bencandy.php?fid=77&id=217479)** **[网址四](https://blog.csdn.net/A_Student_OF_SHANDA/article/details/83507085)** 

很辛苦安装好了，跑了一个yolo网络测试了一下：很nice，截图纪念一下！

![\styles\images\2019-07-23-deep-learning-env-config/2.jpg]({{ '\styles\images\2019-07-23-deep-learning-env-config/2.jpg' | prepend: site.baseurl  }})

![\styles\images\2019-07-23-deep-learning-env-config/1.jpg]({{ '\styles\images\2019-07-23-deep-learning-env-config/1.jpg' | prepend: site.baseurl  }})

![\styles\images\2019-07-23-deep-learning-env-config/4.jpg]({{ '\styles\images\2019-07-23-deep-learning-env-config/4.jpg' | prepend: site.baseurl  }})

![\styles\images\2019-07-23-deep-learning-env-config/3.jpg]({{ '\styles\images\2019-07-23-deep-learning-env-config/3.jpg' | prepend: site.baseurl  }})