# Isaac Sim 串接契約

本 package 不綁定特定 USD scene；只要求 Isaac Sim 透過 ROS 2 bridge 提供兩個同步 RGB topic。

## 相機配置

- `left_camera`：朝危險區入口的左側來向。
- `right_camera`：朝危險區入口的右側來向。
- 兩支相機都應看到等待區、接近區和危險區邊界的一部分。
- 不要讓模型只能看到人臉或手；要保留足以比較行走方向的全身與地面參考。
- 在場景中使用清楚、固定的地面線或區域顏色，使 `APPROACH_ZONE` 與
  `INSIDE_HAZARD_ZONE` 有可見依據。

## ROS 2 Action Graph 必要功能

依 Isaac Sim 版本建立等效 graph，至少包含：

1. playback/tick 觸發來源；
2. ROS 2 context；
3. simulation time/clock publisher；
4. 左相機 render product 與 RGB camera publisher；
5. 右相機 render product 與 RGB camera publisher。

輸出契約：

| 相機 | Topic | Type | Encoding |
|---|---|---|---|
| Left | `/left_camera/rgb` | `sensor_msgs/msg/Image` | `rgb8` 或 `bgr8` |
| Right | `/right_camera/rgb` | `sensor_msgs/msg/Image` | `rgb8` 或 `bgr8` |
| Clock | `/clock` | `rosgraph_msgs/msg/Clock` | simulation time |

兩個影像 publisher 應由同一條 simulation tick 驅動。預設最大 timestamp skew 是
0.20 秒，超過時 VQA node 會等待而不送出不一致的 storyboard。

## 啟動前檢查

```bash
ros2 topic list | grep -E 'left_camera|right_camera|clock'
ros2 topic hz /left_camera/rgb
ros2 topic hz /right_camera/rgb
ros2 topic echo /left_camera/rgb --once --field header
ros2 topic echo /right_camera/rgb --once --field header
```

如果系統一直顯示：

- `waiting_for_both_cameras`：topic 名稱或 ROS domain 不一致；
- `warming_temporal_buffer`：剛開始播放，尚未累積足夠時間；
- `camera_timestamps_not_synchronized`：左右 publisher 沒有共用時鐘/tick；
- `camera_frame_stale`：simulation 停止或 image publisher 未繼續發布；
- `error:model_pipeline`：controller/worker 不可用或模型輸出不符合格式。

## 建議測試情境

每個 scenario 應保存 simulator ground truth，並與 VQA JSON 比較：

1. 左右皆無人；
2. 只從左側接近並進入；
3. 只從右側接近並進入；
4. 接近後停在等待區；
5. 平行路過但不進入；
6. 接近後轉身離開；
7. 左右同時有人；
8. 前景物體遮住其中一支相機；
9. 人已經在危險區內；
10. 快速進入，使 `entry_time=0_TO_2_SECONDS`。

初期應固定相機與光線，先確認問題定義；之後再做相機高度、FOV、照明、衣著、人物數量
與 domain randomization。
