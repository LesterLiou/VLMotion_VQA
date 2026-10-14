# Ground-plane calibration

目前預設是 `future_behavior_policy`，並以 `enable_zone_logic: false` 關閉 zone/TTE。
`trajectory_context_mode: none` 也不會把 ground motion prediction 送入 VQA，因此第一階段
future-behavior 實驗不需要先完成本文件。只有啟用 trajectory context，或把
`decision_mode` 改成 `zone_safety_policy` 時，才需要以下步驟。

影片 demo 的 normalized matrix 只供 UI 與資料流開發。Isaac Sim 或真實相機必須為每個
視角各自求 image footpoint `(u,v)` 到共同 ground frame `(X,Y)` 的 homography。

1. 在 ground plane 放至少四個不共線、覆蓋工作區的已知點；建議 8–12 點。
2. 記錄每點的 ground `(X,Y)`（公尺）和左右影像 pixel `(u,v)`。
3. 用 RANSAC 求 `H_image_to_ground`，不是反方向 matrix。
4. 以未參與估計的點計算 reprojection/ground error。
5. 把 row-major 9 個值寫進 `left_homography`、`right_homography`。
6. 用同一個 ground frame 定義 `hazard_polygon` 與外層 `warning_polygon`。
7. 設 `calibration_mode: metric_homography`；只有誤差通過門檻才設
   `calibration_valid: true`。

對 actor mask 的 ground observation 使用 mask 最下方 4 pixel band 的中位 x 與最低 y，
比 bounding-box center 更接近接地點。遮擋或腳未入鏡時這個假設會失效，因此要用
`tracking_reliable` 與 calibration coverage 過濾。

Isaac Sim 建議同時保存 simulator actor ground truth，以分別量測：footpoint error、
homography error、Kalman velocity error 和 TTE error。不要只量測最後 action accuracy，
否則無法知道錯誤來自哪一層。

若危險區會隨 robot plan 改變，應由 planning node 發佈同一 ground frame 下的未來
swept polygon；目前 config 中的 polygon 是固定值，適合第一階段研究，不代表動態機器人
完整碰撞空間。
