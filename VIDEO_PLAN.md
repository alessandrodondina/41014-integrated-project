# 41014 Integrated Project — Five-Minute Video Plan

## 0:00–0:35 — Introduction
Introduce Track A and the goal: autonomous waypoint navigation and ArUco-guided docking in MuJoCo.

## 0:35–1:15 — Robot and Sensors
Show the MuJoCo environment, obstacles, 36-ray LiDAR and onboard camera.
Explain camera calibration and the known charging-station landmark.

## 1:15–2:00 — State Estimation
Explain differential-drive odometry and EKF prediction/correction.
Show localisation results, estimated pose, ground truth and covariance/uncertainty.

## 2:00–3:25 — Integrated Demonstration
Show the robot following all six waypoints.
Explain proportional feedback, LiDAR safety monitoring and camera-guided docking.
Show the final status and docking distance.

## 3:25–4:15 — Robustness
Compare the three encoder-noise trials.
Explain why the highest-noise trial lost camera tracking.

## 4:15–5:00 — Results and Limitations
Summarise the successful nominal result.
Discuss kinematic simulation, predefined waypoints, marker visibility and potential improvements.

## Recording Checklist
- Show actual code execution and results, not only static screenshots.
- Show estimated state, uncertainty and ground truth together.
- Include at least one varied-condition trial.
- Keep total duration at or below five minutes.
- Upload the video and add its link to README.md and Canvas.
