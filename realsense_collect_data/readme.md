# humanego数据采集
## 环境按照
```bash
pip install -r requirements.txt
```
## 数据采集流程
```
python read_frames.py
```
开始单条视频记录按回车；结束单条视频记录按空格；退出本次记录按esc
说明
* --serial 指定realsense序列号，最开始会进行检测
* --width
* --height
* --fps
* --no-depth 不记录深度
