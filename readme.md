# TACF-Tracker

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2018a](https://img.shields.io/badge/matlab-2018a-yellow.svg)](https://www.mathworks.com/products/matlab.html) |

Matlab implementation of our Towards Robust Visual Tracking for Unmanned Aerial Vehicle with Tri-Attentional Correlation Filters (TACF) tracker.

### Abstract 

>Object tracking has been broadly applied in unmanned aerial vehicle (UAV) tasks in recent years. How-ever, existing algorithms still face difficulties such as partial occlusion, clutter background, and other challenging visual factors. Inspired by the cutting-edge attention mechanisms, a new visual tracking framework leveraging multi-level visual attention to make full use of the information during tracking. Three primary attention, i.e., contextual attention, dimensional attention, and spatiotemporal attention, are integrated into the training and detection stages of correlation filter-based tracking pipeline. Therefore, the proposed tracker is equipped with robust discriminative power against challenging factors while maintains high operational efficiency in UAV scenarios. Quantitative and qualitative experiments on two well-known benchmark with 173 challenging UAV video sequences demonstrate the effectiveness of the proposed framework. The proposed tracking algorithm compares favorably against state-of-the-art methods, yielding 4.8% relative gain in UAVDT and 8.2% relative gain in UAV123@10fps against the baseline tracker while operating at the speed of ∼28 frames per second

### Running instructions

1. Config dataset path in `UAV123_utils/load_video_information.m`,

2. Run `TACF_Demo.m`

### Results on UAV datasets

<details open>
  <summary><b>UAVDT</b></summary>
<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/TACF-Tracker/master/results/overlap_OPE_UAVDT.png" alt="UAVDT_overlap">
</div>
</details>


<details>
  <summary><b>UAV123@10fps</b></summary>
<div align="center">
    <img src="https://raw.githubusercontent.com/vision4robotics/TACF-Tracker/master/results/overlap_OPE_UAV123.png" alt="UAV123@10fps">
</div>
</details>

### Contact 

Yujie He

Email: he-yujie@outlook.com

Changhong Fu

Email: changhong.fu@tongji.edu.cn

### Acknowledgements

We thank the contribution of Dr.`Chen Wang` and Dr. `Martin Danelljan` for their previous work KCC, DSST and ECO. The feature extraction modules are borrowed from the [ECO](https://github.com/martin-danelljan/ECO) and some of the parameter settings and functions are borrowed from [KCC](https://github.com/wang-chen/KCC/tree/master/tracking).