# 雙影片模式

`dual_video_source_node` 將兩個 MP4 當作暫時的左右相機，並發布與 Isaac Sim 模式完全相同的
ROS image topics。因此換成 Isaac Sim 時，VQA prompt、schema 與 decision node 不需修改。

## 預設影片

```text
LEFT  /workspace/example/video/0610_take3n4.mp4
RIGHT /workspace/example/video/0610_take1n2.mp4
```

來源均為 1920×1080、30 FPS。預設輸出為 960×540、10 FPS，減少兩路 raw image 的頻寬。

## 同步方式

- 節點只有一個 ROS timer 和一個共同 `timeline_s`。
- 左右 frame 使用完全相同的 ROS timestamp。
- 每個來源依共同時間選取對應原始 frame。
- 任一影片抵達結尾時，預設兩支一起 rewind。
- 共同有效長度取兩支影片中較短者。

這代表 ROS 資料是同步的，但無法保證兩個獨立錄製檔案中的真人事件原本同步。如果要讓某個動作
在 storyboard 中對齊，調整：

```yaml
dual_video_source:
  ros__parameters:
    left_start_offset_s: 0.0
    right_start_offset_s: 0.0
```

offset 表示從各自影片開頭略過的秒數。

## 啟動

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py
```

影片 demo 預設啟動雙視角 GUI。關閉視窗會結束整個 launch；純 topic 模式使用：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py enable_gui:=false
```

只播放一次：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py loop_videos:=false
```

使用完整解析度時，把設定中的 `output_width` 和 `output_height` 都改成 `0`。完整解析度會明顯增加
ROS raw image 頻寬，但不一定改善 LLaVA 結果，因為 storyboard 最後仍會經過模型 image processor。

## 與 Isaac Sim 切換

影片模式：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py
```

Isaac Sim 模式：

```bash
ros2 launch risk_vqa_sim risk_vqa_sim.launch.py
```

只要 Isaac Sim 也發布 `/left_camera/rgb` 和 `/right_camera/rgb`，其他設定不需要改。
