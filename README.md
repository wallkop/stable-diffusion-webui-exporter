#### 一.导出运行参数
```angular2html
该功能主要是为了快速导出JSON提交到SD后台批量训练。

1.需要先进行一次图片生成, 否则无法导出。
2.点击 [导出-运行参数JSON] 按钮后, 即生成本次运行的JSON。
3.点击下方 [参数下载] 文件区域生成的链接即可下载。
```

#### 二.存档/读档界面参数
```angular2html
该功能主要是为了存档记录当前WebUI的参数, 并在以后进行读档还原。

【如何存档】
1.点击 [存档-界面参数] 按钮, 即生成当前WebUI界面的参数。
2.点击下方 [参数下载] 文件区域生成的链接即可下载。

【如何读档】
1.点击 [读档-界面参数] 按钮, 弹出文件框后选择对应的参数文件。
2.点击 [开始读档] 按钮, 让导入的文件参数生效。

【注意事项】
1.在当前机器存档的文件，也只能在当前机器读档，否则可能因为插件差异导致读档失败
2.如果当前webui变更了插件(包括新增、更新)，则会导致之前保存的文件无法再读档
```

#### 三.目前兼容的插件
* adetailer
* sd-webui-additional-networks
* sd-webui-controlnet
* sd-webui-openpose-editor

#### 四.插件更新日志
* 2023/09/28 `v1.0.0` 第一版插件发布
* 2023/10/10 `v1.0.1` 1.修复了图生图读档失败bug: 当先在文生图中生成一张图片, 再去图生图存档后, 生成的json无法读档 2.修复了图生图的初始图被替换成{{PPP}}标签的问题
* 2023/10/20 `v1.0.2` 修复了不支持导出adetailer中2nd参数的问题
* 2023/10/25 `v1.0.3` 支持导出参数在车间动态切换checkpoint(底模)

#### 五.Bug反馈
* 钉钉联系 `wusilei` 老师
