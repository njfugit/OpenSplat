# OpenSplat 项目详细说明（中文版）

> 本文面向“想重点吃透投影（Projection）与光栅化（Rasterization）”的开发者。会先给全局流程，再把 `Model::forward` 里的渲染主链路逐步拆开到“输入/输出张量级”。

## 1. 项目定位与整体目标

OpenSplat 是一个 C++ 实现的 3D Gaussian Splatting 训练器。它读取外部 SfM/NeRF 项目（COLMAP / Nerfstudio / OpenSfM / OpenMVG）中的相机与稀疏点云，训练得到可渲染的高斯场参数，并导出 `.ply` 或 `.splat` 场景文件。

从系统分层看：

1. **数据层**：统一多种输入格式到 `InputData`（相机 + 稀疏点）。
2. **参数层**：`Model` 维护可学习参数（位置、尺度、旋转、颜色 SH、不透明度）。
3. **渲染层**：投影高斯 -> 球谐着色 -> 光栅化得到图像。
4. **优化层**：基于 `L1 + SSIM` 损失反向传播，并周期性 densify/cull/refine。

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

### 3.2 关键设计点

- **输入统一化**：`inputDataFromX` 自动判别数据源。
- **多分辨率训练**：`getDownscaleFactor(step)` 让前期低分辨率快收敛、后期高分辨率补细节。
- **动态模型容量**：`afterTrain` 按梯度和尺度拆分/复制/删除高斯。

---

## 4. 核心函数调用关系树（细化版）

## 4.1 数据输入与预处理调用树

```text
inputDataFromX(projectRoot, colmapImageSourcePath)
 ├─ inputDataFromNerfStudio
 ├─ inputDataFromColmap
 ├─ inputDataFromOpenSfM
 └─ inputDataFromOpenMVG
```

```text
Camera::loadImage(downscaleFactor)
 ├─ imreadRGB(filePath)
 ├─ 重标定 fx/fy/cx/cy（若图像尺寸与标定不一致）
 ├─ 可选缩放图像和内参
 ├─ 可选去畸变（cv::getOptimalNewCameraMatrix + cv::undistort）
 └─ 更新 Camera 的 K / image / 分辨率参数
```

---

## 4.2 渲染与训练调用树（重点）

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

## 4.3 重点学习：投影（Projection）过程拆解

这一段对应 `Model::forward` 里 **从 3D 高斯到屏幕空间椭圆** 的部分。

### 4.3.1 输入参数（每个 step）

- 几何参数：`means`（中心）、`scales`（log 尺度）、`quats`（旋转）。
- 相机参数：`cam.camToWorld`、`fx/fy/cx/cy`、`height/width`。
- 调度参数：`getDownscaleFactor(step)`（决定当前训练分辨率）。

### 4.3.2 坐标变换链

```text
World Gaussian center (means)
  └─ viewMat (world -> camera)
      └─ projMat (camera -> clip/NDC)
          └─ 屏幕坐标 xys + 投影半径 radii + conic 参数
```

关键点：

1. OpenSplat 会先对旋转矩阵做 `y/z` 翻转以对齐 gsplat 约定。
2. `viewMat` 由 `Rinv` 与 `Tinv` 拼出 4x4 视图矩阵。
3. `projMat` 由视场角（`fovX/fovY`）构造透视投影矩阵。
4. 投影算子输出的 `xys/radii/conics` 是后续光栅化的直接输入。

### 4.3.3 CPU 与 GPU 分歧点

- CPU 路径：`ProjectGaussiansCPU::apply(...)`。
- GPU 路径：`ProjectGaussians::apply(...)` + `tileBounds`。

`tileBounds` 的作用是把屏幕分块，后续光栅化只遍历“被高斯触达”的 tile，减少无效计算。

### 4.3.4 投影阶段输出语义

- `xys`：每个高斯在屏幕空间的中心。
- `radii`：屏幕空间覆盖半径（可视/可裁剪判断的关键）。
- `conics`：2D 椭圆二次型参数（决定像素权重衰减形状）。
- `depths / camDepths`：深度排序或合成时使用。
- `numTilesHit`：GPU 路径用于快速聚合 tile 影响范围。

---

## 4.4 重点学习：光栅化（Rasterization）过程拆解

这一段对应 `Model::forward` 中 **把高斯贡献累积到像素** 的部分。

### 4.4.1 光栅化前的颜色准备

在投影后，先做视角相关着色：

1. 由相机位置构造 `viewDirs`。
2. 用 `degreesToUse = min(step / shDegreeInterval, shDegree)` 动态提升 SH 阶数。
3. `SphericalHarmonics*::apply(...)` 计算每个高斯颜色。

这意味着：训练早期颜色模型更简单（低阶 SH），后期更细腻。

### 4.4.2 光栅化输入

- 几何：`xys`, `radii`, `conics`, `depths/camDepths`。
- 颜色：`rgbs`。
- 不透明度：`sigmoid(opacities)`。
- 背景：`backgroundColor`。
- 分辨率：`height`, `width`。

### 4.4.3 光栅化调用树

```text
if CPU:
  RasterizeGaussiansCPU::apply(xys, radii, conics, rgbs, alpha, cov2d, camDepths, H, W, bg)
else GPU:
  RasterizeGaussians::apply(xys, depths, radii, conics, numTilesHit, rgbs, alpha, H, W, bg)
```

### 4.4.4 光栅化核心直觉

可以把每个像素的颜色看作“多个高斯片元按透明度加权混合”的结果：

- 椭圆高斯给出像素权重（离中心越远权重越低）。
- 深度/排序决定合成顺序（避免前后遮挡错误）。
- alpha 控制该高斯对像素最终颜色的贡献比例。
- 最后与背景色合成，得到 `rgb(H, W, 3)`。

### 4.4.5 与训练稳定性的关系

- 如果 `radii.sum()==0`，`forward` 直接返回背景，避免无效反传崩掉流程。
- `xys.retain_grad()` 让投影结果保留梯度，`afterTrain` 可用它判断“哪些高斯该被细化”。

换句话说：**光栅化不只是“渲染输出”，还是 densify 决策的数据来源之一**。

---

## 4.5 afterTrain（策略中枢）简述

`afterTrain(step)` 会动态调整高斯数量，是训练质量和资源开销的关键平衡器：

1. 累积 `xys` 梯度与可见次数。
2. 对高梯度高斯做 split / dup（densify）。
3. 删除低透明或过大高斯（cull）。
4. 周期性 alpha reset，并清空统计缓存。

---

## 5. 关键数据结构说明

### 5.1 `InputData`

- `cameras`：每个视角的内外参与图像路径。
- `points.xyz/rgb`：稀疏点初始化，直接用于高斯参数初始化。
- `scale/translation`：坐标归一化与回写 CRS 时使用。

### 5.2 `Model` 可学习参数

- `means`：高斯中心位置（N×3）。
- `scales`：对数尺度（N×3，`exp` 后参与投影）。
- `quats`：旋转四元数（N×4）。
- `featuresDc` + `featuresRest`：球谐颜色系数。
- `opacities`：透明度 logit（`sigmoid` 后用于光栅化混合）。

---

## 6. 输出与恢复

- `save(...)`：按后缀决定写 `.ply` 或 `.splat`。
- `loadPly(...)`：支持从历史 PLY 恢复训练，并校验 header 中 iteration 元数据。
- `saveCameras(...)`：输出 `cameras.json`，并可选回到原 CRS 坐标。

---

## 7. 面向“投影/光栅化专项学习”的阅读顺序

1. 先读 `model.cpp` 的 `Model::forward`（从 `getDownscaleFactor` 到 `RasterizeGaussians*::apply`）。
2. 再读 `project_gaussians.*`：理解从 3D 高斯到 2D 椭圆参数。
3. 再读 `rasterize_gaussians.*`：理解 tile、排序、alpha 合成。
4. 最后回到 `afterTrain`：理解为何要保留 `xys` 梯度并据此 densify。

如果你愿意，我下一版可以再给你补一份“逐行对照版”（按 `Model::forward` 的每 10~20 行解释一次张量形状变化）。
