# OpenSplat 项目详细说明（中文版）

> 本文面向“想快速理解 OpenSplat 训练主线 + 核心函数调用关系”的开发者，重点解释训练循环、渲染链路、动态增删高斯（densify/cull）三大核心模块。

## 1. 项目定位与整体目标

OpenSplat 是一个 C++ 实现的 3D Gaussian Splatting 训练器。它读取外部 SfM/NeRF 项目（COLMAP / Nerfstudio / OpenSfM / OpenMVG）中的相机与稀疏点云，训练得到可渲染的高斯场参数，并导出 `.ply` 或 `.splat` 场景文件。

从“系统流”看，它可以拆成四层：

1. **数据层**：统一多种输入格式到 `InputData`（相机 + 稀疏点）。
2. **参数层**：`Model` 维护可学习参数（位置、尺度、旋转、颜色 SH、不透明度）。
3. **渲染层**：投影高斯 -> 球谐着色 -> 光栅化得到图像。
4. **优化层**：基于 `L1 + SSIM` 损失反向传播，并周期性做 densify/cull/refine。

---

## 2. 模块结构速览

- `opensplat.cpp`：程序入口、命令行参数、训练总循环。
- `input_data.cpp/.hpp`：统一输入接口 `inputDataFromX`、相机图像加载与下采样金字塔。
- `model.cpp/.hpp`：训练核心（前向渲染、损失、优化器、densify/cull、存取模型）。
- `colmap.cpp` / `nerfstudio.cpp` / `opensfm.cpp` / `openmvg.cpp`：不同数据源解析。
- `project_gaussians.*`、`spherical_harmonics.*`、`rasterize_gaussians.*`：投影/着色/光栅化算子（CPU/GPU 双路径）。

---

## 3. 主流程（从 main 到输出）

### 3.1 训练主线（高层）

```text
main
 ├─ 解析 CLI 参数（迭代数、分辨率调度、densify 策略等）
 ├─ inputDataFromX(projectRoot)
 │   └─ 自动选择具体数据解析器（COLMAP / Nerfstudio / OpenSfM / OpenMVG）
 ├─ Camera::loadImage(...) 并行加载图像、缩放内参、必要时去畸变
 ├─ InputData::getCameras(validate, valImage)
 ├─ 构造 Model(...)
 ├─ for step in [1..numIters]
 │   ├─ model.optimizersZeroGrad()
 │   ├─ rgb = model.forward(cam, step)
 │   ├─ gt = cam.getImage(model.getDownscaleFactor(step))
 │   ├─ loss = model.mainLoss(rgb, gt, ssimWeight)
 │   ├─ loss.backward()
 │   ├─ model.optimizersStep()
 │   ├─ model.schedulersStep(step)
 │   └─ model.afterTrain(step)   # densify / cull / alpha reset
 ├─ InputData::saveCameras("cameras.json", keepCrs)
 └─ model.save(outputScene, numIters)
```

### 3.2 这一层最关键的设计点

- **输入统一化**：`inputDataFromX` 根据文件特征自动判别数据源，避免用户显式指定 loader。
- **多分辨率训练**：通过 `getDownscaleFactor(step)` 在前期用低分辨率、后期逐步升分辨率，提高训练稳定性和速度。
- **动态模型容量**：`afterTrain` 会根据梯度和尺度拆分/复制/删除高斯，使模型复杂度与场景细节自适应。

---

## 4. 核心函数调用关系树（细化版）

## 4.1 数据输入与预处理调用树

```text
inputDataFromX(projectRoot, colmapImageSourcePath)
 ├─ 若存在 transforms.json
 │   └─ inputDataFromNerfStudio(projectRoot)
 ├─ 若存在 sparse/ 或 cameras.bin
 │   └─ inputDataFromColmap(projectRoot, colmapImageSourcePath)
 ├─ 若存在 reconstruction.json
 │   └─ inputDataFromOpenSfM(projectRoot)
 ├─ 若存在 opensfm/reconstruction.json
 │   └─ inputDataFromOpenSfM(projectRoot/opensfm)
 └─ 若存在 sfm_data.json
     └─ inputDataFromOpenMVG(projectRoot)
```

```text
Camera::loadImage(downscaleFactor)
 ├─ imreadRGB(filePath)
 ├─ 按图像尺寸与标定尺寸差异重标定 fx/fy/cx/cy
 ├─ 可选：按 downscaleFactor 缩放图像和内参
 ├─ hasDistortionParameters ?
 │   ├─ undistortionParameters
 │   ├─ cv::getOptimalNewCameraMatrix
 │   └─ cv::undistort
 ├─ imageToTensor(...)
 └─ 更新 Camera 的 width/height/fx/fy/cx/cy/K/image
```

---

## 4.2 渲染与训练调用树（核心）

```text
Model::forward(cam, step)
 ├─ getDownscaleFactor(step)
 ├─ 构建 viewMat/projMat
 ├─ ProjectGaussiansCPU::apply 或 ProjectGaussians::apply
 │   └─ 产出 xys/radii/conics/(depths,numTilesHit 等)
 ├─ 若无可见高斯：直接返回 backgroundColor
 ├─ SphericalHarmonicsCPU::apply 或 SphericalHarmonics::apply
 │   └─ 基于视角方向 + SH 系数得到每个高斯颜色
 ├─ RasterizeGaussiansCPU::apply 或 RasterizeGaussians::apply
 │   └─ 把高斯混合到像素网格，输出 rgb
 └─ clamp 到 [0,1]
```

```text
训练单步（main 循环中的一次 step）
 ├─ model.optimizersZeroGrad()
 ├─ rgb = model.forward(cam, step)
 ├─ gt = cam.getImage(model.getDownscaleFactor(step))
 ├─ loss = model.mainLoss(rgb, gt, ssimWeight)
 │   ├─ l1(rgb, gt)
 │   └─ ssim.eval(rgb, gt)
 ├─ loss.backward()
 ├─ model.optimizersStep()
 ├─ model.schedulersStep(step)
 └─ model.afterTrain(step)
```

---

## 4.3 afterTrain（最核心策略）详细拆解

`afterTrain(step)` 是 OpenSplat 训练效果与效率的“策略中枢”。它并不是简单的优化器 step 后处理，而是会动态改变参数张量长度（即高斯数量）。

### 阶段 A：统计可见高斯梯度与屏幕尺寸

- 仅在 `step < stopSplitAt` 时统计。
- 使用 `xys.grad()` 的 L2 范数作为“该高斯对当前损失敏感度”的 proxy。
- 对每个高斯累计：
  - `xysGradNorm`：梯度和
  - `visCounts`：被看见次数
  - `max2DSize`：历史最大屏幕占比（由 `radii` 推算）

这一步等价于构建“哪些高斯值得加细节”的证据。

### 阶段 B：densify（拆分 + 复制）

在 `step % refineEvery == 0 && step > warmupLength` 时触发 refinement 窗口；且在满足周期条件时进入 densification：

1. 计算平均梯度：`avgGradNorm = xysGradNorm / visCounts`。
2. 得到高梯度掩码 `highGrads`。
3. 对高梯度中“尺度大”的高斯做 **split**：
   - 在局部尺度坐标采样噪声；
   - 由四元数旋转到世界方向；
   - 生成新的 `splitMeans`；
   - 子高斯尺度缩小（`sizeFac`）；
   - 颜色/不透明度/旋转按父高斯复制。
4. 对高梯度中“尺度小”的高斯做 **dup**（直接复制）。
5. 把新增高斯拼接到参数张量末尾，并调用 `addToOptimizer(...)` 扩展 Adam 状态张量（`exp_avg` / `exp_avg_sq`），确保优化器与新参数同维度。

### 阶段 C：cull（删除低价值高斯）

- 先按 `sigmoid(opacities) < cullAlphaThresh` 删除“近似透明”高斯。
- 若经历一定训练阶段，还会删“过大高斯”（世界尺度太大或屏幕占比过大）。
- 删除时调用 `removeFromOptimizer(...)` 同步裁剪 Adam 状态。

### 阶段 D：alpha reset（周期性重置透明度）

- 在特定 refinement 周期把 opacity 上限压到阈值附近。
- 同时重置 opacity 优化器动量，避免历史动量把透明度立即拉回旧状态。

### 阶段 E：清理状态

- 清空本轮统计缓存 `xysGradNorm/visCounts/max2DSize`。
- GPU 模式下尝试清空缓存分配器，减少显存峰值压力。

---

## 5. 关键数据结构说明

### 5.1 `InputData`

- `cameras`：每个视角的内外参与图像路径。
- `points.xyz/rgb`：稀疏点初始化，直接用于高斯参数初始化。
- `scale/translation`：坐标归一化与回写 CRS 时使用。

### 5.2 `Model` 可学习参数

- `means`：高斯中心位置（N×3）。
- `scales`：对数尺度（N×3，使用 `exp` 还原）。
- `quats`：旋转四元数（N×4）。
- `featuresDc` + `featuresRest`：球谐颜色系数。
- `opacities`：透明度 logit。

这些参数分别由独立 Adam 优化器管理，便于不同学习率策略。

---

## 6. 输出与恢复

- `save(...)`：按后缀决定写 `.ply` 或 `.splat`。
- `loadPly(...)`：支持从历史 PLY 恢复训练，并校验 header 中的 iteration 元数据。
- `saveCameras(...)`：输出 `cameras.json`，并可选回到原 CRS 坐标。

---

## 7. 给二次开发者的切入建议

1. **先读主链路**：`opensplat.cpp` -> `Model::forward` -> `Model::afterTrain`。
2. **再看输入解析器**：按你常用数据源重点读对应 `xxx.cpp`。
3. **最后看算子实现**：若要做性能优化，再深入 `project/rasterize/sh` 的 CPU/GPU 内核。

如果你要改效果，优先调 `afterTrain` 中的 densify/cull 策略；如果你要改速度，优先看投影/光栅化算子和分辨率调度。
