# Hero 检测 PoC：HOG + Linear SVM

一个隔离的实验，只回答三个问题：**能不能稳定找到 Hero、够不够快、误检能不能接受**。
**不接入 production**：不改 `src/vehicle.py`、不改 `src/red_arrow.py`、不改导航/API/相机。

## 三个脚本

```zsh
# ① 采数据（拖框标注）
python tools/hero_lab/hero_capture.py --note sess1
#   拖框住英雄车 → SPACE 保存 → S 跳过 → Q 退出
#   每张图自动在同一帧别处采 4 个负样本（其它车/空位/道路/LED/闸机/边缘）

# ② 训练（按采集批次切 train/val，避免同段视频泄漏）
python tools/hero_lab/train_hog.py

# ③ 实时检测（打印 FPS / 帧时间 / 命中）
python tools/hero_lab/live_detect.py --seconds 20
python tools/hero_lab/live_detect.py --seconds 20 --show    # 画框
```

## 设计（为什么是这些数）

| 项 | 值 | 理由 |
|---|---|---|
| 窗口 | **64×64** | 量过的 Hero bbox 约 90×40（含车顶标记）→ 约 2:1；64 能被 8×8 cell 整除 |
| HOG | cell 8×8、block 2×2、stride 8、9 bins、L2-Hys | OpenCV 常规默认值，不调 |
| 特征长度 | **1764** | 一个窗口一次点积 → 推理 <0.1 ms |
| 分类器 | `cv2.ml.SVM_LINEAR` | 轻量，导出为 `hero_hog.yml` |
| 搜索 | ROI = 停车场有效区（去掉墙/货架）+ 2 个尺度 × stride 24 | 先控成本 |
| 跟踪 | 上一帧找到 → 只在 ±120 px 内搜；丢失 → 全 ROI | 借鉴 production 的 `prefer_near`，代码独立 |

## 已经测到的成本（还没有真实数据，只测了速度）

```
近区搜索（有上一帧位置）: 10.8 ms/帧  →  92 FPS  ✓
全图搜索（冷启动/丢失一次）: 135 ms/帧  →  7 FPS  ✗
每帧一次 grab（HTTP MJPEG）另计
```

→ **结论**：速度上"有跟踪"能满足 ≥20 FPS；"每帧全图搜"不行。这符合预期（production 里的
模板匹配也是靠 ROI 才跑得动）。

## 数据布局（全部在 `tools/hero_lab/data/` 内）

```
frames/000123.jpg     整帧
boxes.jsonl           每行: frame, bbox, negatives, note(采集批次), at
pos/000123.jpg        正样本（缩放到 64×64）
neg/000123_0.jpg…     自动负样本
```

## 待做（需要真实标注）

- [ ] 采 80~150 张正样本，覆盖：全桌 x/y、0~360° 朝向、中央/边缘/四角、清晰/轻微模糊、不同亮度、**其它颜色车辆靠近 Hero**
- [ ] `--note` 每批不同，保证 train/val 按批次切
- [ ] 跑 `train_hog.py`，记录 recall / precision
- [ ] `live_detect.py --seconds 60`，记录平均 FPS / p95 / 命中
- [ ] 用**未参与训练**的一段视频做最终验证
- [ ] 失败案例各存一张：miss / false positive / blur / edge / rotation
- [ ] 写成 benchmark 报告

## 验收线

```
≥ 20 FPS（近区） + recall ≥ 95% + 误检很少  →  值得继续
< 10 FPS，或边缘/旋转大量漏检，或其它车频繁误检  →  判定不适合，不再堆补丁
```
