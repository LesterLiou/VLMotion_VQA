# X9 相機跨電腦訂閱

## RViz 直接看影像

使用本專案 `ros2_ws/src/future_risk_vqa/image/` 內的檔案；可將此資料夾複製到另一台電腦。關閉原先的 RViz，於此資料夾執行：

```bash
sudo apt install ros-humble-compressed-image-transport
bash view_x9_cameras.sh
```

若解碼套件已安裝，可略過 apt 指令。此啟動器在同一程序環境載入 DDS 設定，
使用三個 `rviz_default_plugins/Image` 顯示訂閱 JPEG topics。
Humble 的 Image display 根據 topic 的 `/compressed` 後綴選擇 image_transport 解碼器。
使用 Best Effort、Volatile、depth 1。純 2D Image 不需要相機 TF；
Camera display 用於 3D 疊合，會額外依賴相機參數與 TF。
若出現 Fixed Frame 警告，可先看 Image 面板自身的 Topic/Image 狀態。

RViz 已在執行時，於其他終端 source 設定不會更新那個 RViz 的環境。
歷史本機檢查曾確認 JPEG 可以解碼（本次整理未重測網路／相機）；尚未驗證對端 RViz 接收。

若顯示 `No Image`，在同一個已 source 連線設定的終端執行下文 JPEG echo 指令：
有 header 表示 DDS 影像訊息有到達該電腦，接著查 RViz 外掛／顯示設定；
沒有 header 則繼續查跨電腦 DDS 資料傳輸。

RViz 實作參考：[Humble ImageTransportDisplay](https://github.com/ros2/rviz/blob/humble/rviz_default_plugins/include/rviz_default_plugins/displays/image/image_transport_display.hpp)。

本次在 Isaac Sim 電腦（192.168.0.228）直接訂閱了目前播放中的場景：
三路 `image_raw`、三路 `camera_info` 與 `/joint_states` 都有資料。
原始場景的 `/World/ActionGraphCamera` 啟用且相機路徑正確，本次沒有修改 USD。

「對端 topic list 看不到 camera」是探索問題；topic 名稱出現後但沒有畫面，
才進一步檢查傳輸、訊息類型、QoS 或影像大小。尚未在另一台電腦驗證成功。

## 在另一台電腦執行

將 `connect_x9_cameras.sh` 和 `check_camera_topics.py` 複製到兩台電腦，並在兩邊的
terminal 都載入相同設定：

```bash
source ./connect_x9_cameras.sh
ros2 daemon stop
ros2 topic list -t --no-daemon --spin-time 10
/usr/bin/python3 check_camera_topics.py --seconds 10
```

腳本指定 domain 0、允許非本機 ROS 通訊，並使用 Fast DDS 的自動 multicast
discovery，不綁定任何電腦 IP。因此 DHCP 位址變更時不必修改腳本。兩台電腦仍須位於
同一個允許 multicast 與用戶端互通的 LAN；訪客 Wi-Fi、AP/client isolation、不同 VLAN
或防火牆阻擋 DDS UDP 時，仍無法自動探索。

先確認小型的相機參數訊息：

```bash
ros2 topic echo /camera/front/color/camera_info sensor_msgs/msg/CameraInfo \
  --once --no-daemon --qos-reliability best_effort
```

再確認 JPEG 影像（只列 header，避免列出大量二進位資料）：

```bash
ros2 topic echo /camera/front/color/image_raw/compressed sensor_msgs/msg/CompressedImage \
  --once --no-daemon --qos-reliability best_effort --field header
```

原始影像的 topic 是 `/camera/front/color/image_raw`，類型 `sensor_msgs/msg/Image`。
左右相機將 `front` 換成 `left` / `right` 即可。
JPEG topic 使用 `sensor_msgs/msg/CompressedImage`，需以對應類型訂閱或使用
支援 compressed image_transport 的顯示程式。

## 本機修復與實測

原有 `isaac-camera-relay.service` 使用的
`~/.config/isaac-ros/camera-fastdds.xml` 將 `maxMessageSize` 設為 1400，
導致轉發器收不到 Isaac 的原始影像。在相同其他設定下，6 秒測試：
1400 收到 0 張，65500 收到 68 張。
已備份 XML、改成 65500 並重啟這個轉發服務；沒有重啟 Isaac Sim。

修復後 8 秒的本機檢查：前／左／右原圖收到 75／82／69 張，
JPEG 收到 40／40／36 張，關節收到 198 筆。當時紀錄提及 `camera_lan_local_verification.json`，目前 checkout 沒有此產物，不能視為可重現證據。
原圖每張 921600 bytes；該次 JPEG 樣本約 6.6～47.2 KB，適合先測 Wi-Fi。

若更新對端設定後依然看不到相機，請保留 `topic list` 與檢查程式輸出，
再查對端 DDS/firewall/AP 路徑；不需先更改場景 Graph。

參考：[Fast DDS initial peers](https://fast-dds.docs.eprosima.com/en/2.x/fastdds/use_cases/wifi/initial_peers.html)、
[ROS 2 QoS 相容性](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)。
