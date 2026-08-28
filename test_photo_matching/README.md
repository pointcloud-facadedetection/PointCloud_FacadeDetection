# 独立图片匹配测试

这个程序只读取两张普通图片，使用 SuperPoint 提取关键点、LightGlue
进行特征匹配，不依赖点云或现有项目数据。

## 运行

在项目根目录执行：

```powershell
pip install -r .\test_photo_matching\requirements.txt
python .\test_photo_matching\app.py
```

## 操作

1. 上传 Image A。
2. 可选：点击“框选目标建筑（GrabCut）”，在弹出窗口中框住目标建筑，
   按 Enter 或 Space 确认。分割完成后可以用“使用 GrabCut”随时切换
   原图和分割图匹配。
3. 上传 Image B。
4. 调整匹配置信度下限；数值越大，保留的匹配越可靠、数量通常越少。
5. 使用“显示 matches”和“显示 keypoints”复选框切换可视化内容。
6. 点击“随机换 10 个”查看另一组匹配。

程序优先使用 CUDA，CUDA 不可用时自动改用 CPU。首次运行可能需要下载
SuperPoint 和 LightGlue 模型权重。为保证速度和显存占用，任一边超过
1600 像素的图片会按比例缩小后再进行匹配。
