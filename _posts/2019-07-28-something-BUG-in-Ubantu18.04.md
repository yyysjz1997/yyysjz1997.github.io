---
layout: post
title:  遇到Ubantu（V18.04）的BUG
date:   2019-07-28 18:30:00 +0800
categories: 技术交流
tag: 软件领域
---

* content
{:toc}

刚装好了Ubantu的最新（稳定）版本18.04，设置默认的输入法为英文+中文智能拼音。结果在使用中文智能拼音输入法的时候，遇到了这样奇怪的BUG。。。

![\styles\images\2019-07-28-something-BUG-in-Ubantu 18.04/1.jpg]({{ '\styles\images\2019-07-28-something-BUG-in-Ubantu 18.04/1.jpg' | prepend: site.baseurl  }})

只要一调到中文智能拼音输入法的时候，就会一直出现图中的字体，在输入法设置中，也根本关不掉，不知道为什么。。。。而且系统自带的输入法中，只有这个是中文的拼音输入法。

**解决方法：**

毕竟Ubantu是开源的Linux系统，而且可能我在安装Ubantu系统的时候可能出现了某些问题。所以我直接把这个输入法删除，直接在terminal中sudo安装了搜狗的输入法。直截了当，问题消失~

反正是能用了，自己也没有再深究原因。。。