# stable-diffusion-webui-exporter

### WebUI-参数管理插件使用说明

#### 一.导出运行参数
```angular2html
该功能主要是为了快速导出JSON提交到SD后台批量训练。

1.需要先进行一次图片生成, 否则无法导出。
2.点击 [导出-运行参数JSON] 按钮后, 即生成本次运行的JSON。
3.点击下方 [参数下载] 文件区域生成的链接即可下载。
```

#### 二.导出/上传UI参数
```angular2html
该功能主要是为了保存记录当前WebUI的参数, 并在以后进行导入还原。

【如何导出】
1.点击 [导出-UI参数文件] 按钮, 即生成当前WebUI界面的参数。
2.点击下方 [参数下载] 文件区域生成的链接即可下载。

【如何上传】
1.点击 [上传-UI参数文件] 按钮, 弹出文件框后选择对应的参数文件。
2.点击 [确认上传] 按钮, 让导入的文件参数生效。
```
