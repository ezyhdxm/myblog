---
layout: post
title: "我实际上是如何创建我的博客的"
comments: true
date:  2021-08-13 08:13:00 +0800
tags: random
lang: zh
---



<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
---



### 1. Introduction

我建了一个博客。有的小朋友就问我：“博客要怎么建呀？” 于是我来具体讲讲我是怎么建的，当然也是为了以后自己能有个参考。我采用的是 Jekyll + GitHub Pages 的搭建模式。这么做主要有两个原因，一来 GitHub Pages 是免费的，适合穷学生；二来就要说说我是怎么心血来潮开始搭博客的了。

话说一切至少要追溯到 2020 年的四月份。那段时间正是疫情，大家都待在家里写着毕业论文。我想看些深度学习相关的东东，于是整天在谷歌上翻来翻去。大约是某天看 VAE 还是 GAN 之类的东东时，发现了 Lilian Weng 的[博客](https://lilianweng.github.io/lil-log/)。把自己学的东西写博客上，这个想法对我很有吸引力。当然，之前也看过谢益辉大人的[博客](https://yihui.org/)，觉得可以值得效仿，可惜那时搞不明白他是怎么建站的。有了这两个激励，加上在家时间也比较充裕，我就开始捣鼓了。我厚颜无耻地将 Lilian Weng 的 GitHub 仓库整个搬了下来，这就开始了魔改之路。由于她用的是Jekyll + GitHub Pages 的模式，于是我也就采用了这个模式。

那么在开始之前，我先说说我会用到哪些工具。我一般会把文章写成 Markdown 的格式，所以在写文字内容的时候，我一般会先用 [Typora](https://typora.io/) 来写。如果我想加入更多 HTML 元素，那么我会用 [VS Code](https://code.visualstudio.com/) 来处理。当然，我们做许多事情都离不开 [Git](https://git-scm.com/) 这个工具，我主要是用里面的 Git Bash，在这里面可以执行 Linux 的那些指令，下面给小朋友列几个常用的指令吧

-  `ls`：查看指定目录下包含哪些文件，效果如下

  ![ls](/hao-blog/assets/images/2021_08_13/image-20210813094943294.png)

   

- `cd`：切换目录需要使用。在路径中，`.` 表示的是当前目录，而 `..` 表示上级目录，效果如下

  ![cd](/hao-blog/assets/images/2021_08_13/image-20210813095228653.png)



- `mkdir`：新建文件夹。用法为 `mkdir <filename>`，例如

  ![mkdir](/hao-blog/assets/images/2021_08_13/image-20210813095649142.png)



感觉会这几个就已经足够建站用了。想要认真学习的话可以看看[这里](https://missing-semester-cn.github.io/)。



### 2. GitHub Pages

接下来我们讲一下怎么来弄 GitHub Pages. 它是一个静态网站托管服务，这意味着什么我也不知道， ：）不过现在咱也不关心这个。只需要注意一个点，就是它的仓库空间有 1G 的大小限制，所以不要往上面丢太多东西啦。如果有小朋友还没有 GitHub 账号，那不妨先弄一个。弄好以后，我们可以跟着它的[官方教程](https://pages.github.com/)来操作。教程写得很清晰，这里我就不赘述了。需要注意的一点是你的域名会是***username\*.github.io**，开头的那个必须是你的用户名。这也就是说大家要先给自己起一个好用户名哦。



### 3. Jekyll

接下来讲讲我们的 Jekyll. 它是一个用来把文本文件转换为静态网站的工具。在装它之前，大家需要先装一些必要的东东。这些东东可以在[这里](https://jekyllrb.com/docs/installation/)看到。接下来按照[这里](https://jekyllrb.com/docs/)的步骤去做就好啦。当然，如果你已经有了一个配置好的文件夹，那么每次做完修改以后只要执行里面的第五、六步就可以预览效果了。操作上可以参考下面的图片

![jekyll](/hao-blog/assets/images/2021_08_13/image-20210813105329147.png)



在 Git Bash 中输完指令后就可以在浏览器中进行预览了！

![preview](/hao-blog/assets/images/2021_08_13/image-20210813105414630.png)



---



### 4. 网站文件结构

我的网站在结构上大概是这样子的

```
.
├── _data
│   ├── navigation.yml
├── _includes
│   ├── head.html
│   ├── header.html
│   └── ...
├── _layouts
│	├── default.html
│	├── post.html
│	└── ...
├── _posts
│	└── ...
├── _sass
│	└── ...
├── assets
│	├── css
│	├── fonts
│	├── images
│	└── ...
├── tag
│	├── random.md
│	└── ...
│
...
```

这里稍微解释一下各个部分是干什么的。`_data`那个就先不用管了；`_includes` 里面主要放一些每个页面都会用到的 HTML 文件，这里也不详细解释了；`_layouts` 大概是放各个页面统一的格式 ；`_sass` 里主要是 .scss 文件，主要起修饰外观的作用；`_posts` 就是我们放博客文章的地方了，待会详细说说；`assets` 用来放网页里用到的各种图片、音频、字体之类的，主要起到方便管理的作用；`tag` 用来存放标签，这个后面会说一说。重要的大概就是这些了。



### 5. 一篇博客的诞生

假设大家已经把所有东东都配置好了，那我们就来写博客吧！在博客的最开头，我们需要加一段 [YAML 格式](https://zh.wikipedia.org/wiki/YAML)，通常来说这在 VS Code 里面进行添加会比较合适。以这篇文章为例，它的格式如下

```yaml
layout: post
title: "我实际上是如何创建我的博客的"
comments: true
date:  2021-08-13 08:13:00 +0800
tags: random
lang: zh
```

`layout` 明确了这是一篇博客，所以采用 `post` 的格式。这里 `post` 就对应了 `_layouts` 文件夹下的 `post.html` 文件。`title` 就是文章的标题。`comments` 对于了这个页面是否可以评论，要是置为 `false` 就是所谓的关评论了呵呵。`date` 我一般写为开始写作的日期。`lang` 是语言设置，不同语言字体加载可能会不一样，所以可以设置设置，这里设置为中文。`tags` 是文章的标签，它能让我们显示一个这样的东东：

![tag](/hao-blog/assets/images/2021_08_13/image-20210813115020801.png)

 然后所有的标签还可以在[标签页](https://ezyhdxm.github.io/hao-blog/tags.html)看到，方便大家按类别取用咯。当然，每个标签需要在 `tag` 文件夹中编写一下，格式很简单。以 random 标签为例，random.md 实际上就长这样

```yaml
---
layout: tagpage
title: "Tag: random"
tag: random
---
```

它里面就只有一个 YAML 格式。想添加新的标签就在 `tag` 文件夹中添加吧！

在 YAML 格式的下方，我们可以考虑给文章插入一段目录。这是由下面这行代码实现的

```
{: class="table-of-content"}
* TOC
{:toc}
```

这一段最好也在 VS Code 中进行添加。这以后就到正文啦！



#### 5.1 插入图片

写文章需要图文并茂。于是插入图片就成了一个常规操作。在 Markdown 中，我们可以使用 `![]()` 的格式来插入图片。`[]` 中放的是对图片的描述文字，当图片加载失败的时候就会显示；`()` 里放的是图片的地址。我们举个例子，比如说我们要插入一张 NVidia 最新 GAN 的演示动图

![gan](/hao-blog/assets/images/2021_08_13/gan.gif)

我会将它放在 `assets/images/2021_08_13/` 中， 然后用绝对路径来找到这张图，代码就是 `![gan](/hao-blog/assets/images/2021_08_13/gan.gif)`. 大家可以根据自己的实际情况来修改代码哦。



#### 5.2 嵌入视频

如果想要播放视频的话，我们一般不会把视频丢到 GitHub Pages 的库里，那样太占地方啦！我们一般会把它上传到其它服务器中，然后再嵌入我们的博客中。假设我们想嵌入一个 b 站视频，比如说[这一个](https://www.bilibili.com/video/BV1cA411j7uL?from=search&seid=5476651152876306862)。那么我们可以去到 b 站，点击转发按钮，复制嵌入代码，

![share](/hao-blog/assets/images/2021_08_13/image-20210813122750392.png)

然后直接粘贴到 Markdown 文件中对应位置就可以了！油管视频也是一样操作哦。

<div style="text-align: center;">
	<iframe src="//player.bilibili.com/player.html?aid=330079542&bvid=BV1cA411j7uL&cid=251106764&page=1" 		scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="428" height="300"> </iframe>
</div>

想要视频居中的话就在复制的代码两头加上下面的代码（即把 iframe 放在下面两行之间）就可以实现居中了。想调整大小之类的就用 `width`、`height` 之类的参数去调整吧！这谷歌一下就可以找到答案啦！

```html
<div style="text-align: center;">
</div>
```



#### 5.3 嵌入音频

想要放音频的话也是可以的哦！最好的方法应该还是使用 iframe，但是我在网易云那里遇到了麻烦。其它地方还可以考虑 SoundCloud，然而它超限制版权；还有 Spotify ！不过我之前没用...... 它的效果如下

<iframe src="https://open.spotify.com/embed/playlist/3Dh4Qk3vJIK3Oa92Pw54vz" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>

如果想上传的音频文件只有自己有的话怎么办呢？那就类似于放图片，只不过我们也得要写直接 HTML. 我手上有一首很久以前写的歌，现在我要把它挂上博客，可以先将它放到对应的文件夹中，然后使用下面这行代码

```html
<audio ref='themeSong' src="/hao-blog/assets/audios/How/Dark.mp3" controls></audio>
```

<audio ref='themeSong' src="/hao-blog/assets/audios/How/Dark.mp3" controls></audio>



要说的应该就是这些啦！



### 6. 上传 GitHub Pages

当我们写好一篇博客后，就可以将它提交到 GitHub Pages 上让所有人都看见啦！我们还是回到放所有文件的那个文件夹，在那里输入 Bash 指令

```bash
git add .
```

然后继续输入指令创建一个新提交。`-m` 后面可以添加一些提交信息。

```bash
git commit -m "SOME COMMIT MESSAGE"
```

最后传送到远端。以我的为例，就是

```bash
git push -u origin gh-pages
```

这样就成功了！在我这里看起来是这样子的哦。

![git](/hao-blog/assets/images/2021_08_13/image-20210813132711379.png)



### 7. 做个总结

整个流程粗略来讲就是这样。我水平超级菜的，做法肯定不是最好的，甚至不是合理的，仅提供一个参考。如果大家有问题，欢迎讨论哦，不过也建议大家去看看[这篇东东](https://github.com/ryanhanwu/How-To-Ask-Questions-The-Smart-Way/blob/main/README-zh_CN.md)。就这样啦。





