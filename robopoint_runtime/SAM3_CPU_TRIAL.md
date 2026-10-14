# SAM3 CPU trial — 2026-09-17

Changed only tracker device from cuda to cpu in both future_risk_isaac.yaml and future_risk_video_demo.yaml. Copied these configs into the existing ROS install/share configuration directory (not symlinked here). Existing controller and 4-bit 13B GPU worker were retained.

Ran the existing video demo without GUI, OMP_NUM_THREADS=4 MKL_NUM_THREADS=4, bounded to 100 seconds. All other pipeline parameters stayed as configured, including floor segmentation and per-track VQA.

Evidence: results/future_risk_vqa/20260917_200959_sam3_cpu_smoke/events.jsonl

First CPU cycle: inference_ms=57014.444; detected L1,L2,R1,R2. CPU tracker RSS snapshot ~4.5 GiB; total GPU memory snapshot 9281 MiB, with no SAM3 GPU process. VQA was reached, but its report was UNAVAILABLE: "model response does not contain an object". This is distinct from the earlier CUDA OOM and is not a successful end-to-end semantic result. CPU latency also makes existing temporal synchronization/timeout settings unsuitable for real-time operation.

Start the same GUI demo with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ros2 launch future_risk_vqa future_risk_video_demo.launch.py model_name:=robopoint_vqa_only_317677
```

Run inside the existing inference container after sourcing ROS and the workspace. The controller and trained worker must be running. To revert, change sam3_person_tracker.ros__parameters.device to cuda in the two source configs and rebuild/install the package or explicitly pass config:=/workspace/ros2_ws/src/future_risk_vqa/config/<filename>.
