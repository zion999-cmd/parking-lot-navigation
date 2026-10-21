# 论文量化证据：只读审计 + 可重复 benchmark 设计

审计日期 2026-10-20。**本文不含任何推测数据**：凡是数字，都注明它来自哪个文件、哪次运行。

> **未测试项，明确声明**：RFID、地磁传感器**尚无硬件**，本项目**从未测试**，论文中不得出现相关数据。

---

## A. Hero 视觉

### 现在能直接拿到的

| 指标 | 能否直接取得 | 出处 | 实测值（真实运行） |
|---|---|---|---|
| **vision loop FPS** | ✅ | `run/vision.log` 每行 `fps=` | n=39704，中位 **13.8**，p95 **38.5**，max 49.8 |
| **PC source FPS** | ✅ | 同上 `src=` | n=39704，中位 **17.7**，p95 **27.7**，max 27.8 |
| **centre x/y** | ✅ | `vision/out/parking_state.json` → `vehicle.x/y` | 实时可读（例：720, 320） |
| **heading / front_heading** | ✅ | 同上 → `vehicle.heading`、`vehicle.front_heading` | 实时可读（例：94.2°） |
| **visible（每帧是否检测到）** | ✅ | 同上 → `vehicle.visible` | 实时可读 |
| **detection FPS / frame ms** | ⚠️ **仅 HOG PoC 内有** | `tools/hero_lab/` 的 benchmark | **4.8 ms / 209 FPS（离线 PoC，见 §A2）**；**production 内没有单独计时** |
| **confidence / score** | ❌ **production 无** | — | `VehiclePose.as_dict()` 不输出 margin；`hero_hog.detect()` 内部有 `margin`，但没写进契约 |
| **TP / FN / FP 统计** | ❌ **无** | — | 日志里没有任何 per-frame 检测或标注比对 |

### A2. 已经做过的、可引用的 Hero 检测实验

来源：`tools/hero_lab/BENCHMARK_HOG.md`，2026-10-20 实测。

```
数据集     124 正样本（人工拖框）+ 496 网格负样本 + 744 难负样本
采集       3 个批次（sess1 42 / sess2 46 / sess3 36），位置 x 23~753、y 85~513
切分       按批次整批切，三轮交叉验证
分类器     cv2.HOGDescriptor(8×8 cell, 2×2 block, stride 8, 9 bins) + cv2.ml.SVM_LINEAR
```

| 验证批次 | recall | precision | 漏检 | 误检 |
|---|---|---|---|---|
| sess1 | 0.976 | 1.000 | 1 | 0 |
| sess2 | 0.978 | 1.000 | 1 | 0 |
| sess3 | 1.000 | 1.000 | 0 | 0 |
| **合计** | **0.984** | **1.000** | **2 / 124** | **0 / 496** |

```
实时（30 s 连续，含 HTTP 取帧）: 202/202 命中，avg 4.8 ms，p95 5.4 ms，209 FPS
两步法（HOG 出框 + 框内红箭头定向）: 6 帧内 heading 93.5~95.6°（±1°）
```

**这批数字的适用边界（必须写进论文）**：
- 单一场景、单套灯光、单次布置；**不是跨天/跨场地**
- 负样本来自网格采样 + 模型自身误检，**没有系统性覆盖"半个车/被遮挡/车影"**
- **124 个正样本** → 98% 置信区间约 **±3%**

### A3. 缺失项的最小补测方案

新增 **一个** 只读脚本（不进 production）：

```
tools/bench/hero_bench.py
  输入: 一段录像（文件或 relay 抓的帧序列）+ 每帧的 ground truth（见 B）
  做法: 逐帧调用现有的 vision/src/hero_hog.py + red_arrow.detect_in_box()
        记录 per-frame: visible, x, y, heading, margin, 处理耗时
  输出: CSV（每帧一行）+ 汇总 JSON
```

**要注意**：`margin` 目前**没有**写进 `parking_state.json`。补测脚本可以直接调 `hero_hog.detect()`
拿 `margin`，**不需要改 production**。

---

## B. 视觉定位精度

### 现状

❌ **没有**任何"人工 ground truth vs 系统输出"的比对工具，也**没有** `x_gt/y_gt/heading_gt` 的记录格式。

### 最小方案

1. **标注**：复用现成标注工具的"英雄车框"模式（`tools/label_ui.html` 的 hero 模式）标 `(x_gt, y_gt)`；
   `heading_gt` 用一个**量角器贴纸**或在桌上画基准线人工记录（**不需要新工具**）。
2. **脚本**：`tools/bench/pose_error.py`
   ```
   输入: CSV(帧名, x_gt, y_gt, heading_gt) + 同一时刻的 parking_state.json 快照
   输出: position_error_px / position_error_cm / heading_error_deg
         mean / median / p95 / max
   ```
3. **px→cm 换算**：代码里已有标尺常量 `PX_PER_M = 20.0`（`vision/web/navigation.html:114`）
   → **1 px = 5 cm**。论文里要写明这是**假设值**，建议实测一次（量一个车位宽度 ÷ 它的像素宽度）。
4. **heading 误差**：必须用**环形角度差** `|((a-b+180) mod 360) - 180|`，不能直接相减。

**样本量建议**：≥20 个位置 × 3 个朝向 = 60 个点；覆盖中央/边缘/四角。

---

## C. 48 车位 occupancy

### 现在能直接拿到

| 项 | 能否直接取得 | 出处 |
|---|---|---|
| 每车位状态 | ✅ | `parking_state.json → spots[id].state` = `EMPTY / OCCUPIED / UNKNOWN` |
| 每车位 `changed_ratio` | ✅ | 同上（阈值 0.30，`occupancy.py:68`） |
| **人工真值 vs 系统输出的 confusion matrix** | ❌ **无** | — |

### 最小方案

`tools/bench/occupancy_bench.py`
```
输入: 人工真值表 CSV(车位id, 真值) + 同时刻的 parking_state.json
     还可以直接从 48 个车位的现场照片人工判读（不需要新硬件）
输出:
      confusion matrix（3×3，含 UNKNOWN）
      accuracy
      false occupied（真值空、系统报占用）—— 最危险的一类
      false empty（真值占用、系统报空）
      unknown rate
```
**必测场景**（各 ≥3 次）：全空、单车、多车、车压线、光照变化。
**现状快照**（仅说明格式可用，**不是精度结论**）：`EMPTY 39 / OCCUPIED 9 / UNKNOWN 0`。

---

## D. Navigation（route planner）

### 现在能直接拿到

| 项 | 能否直接取得 | 出处 |
|---|---|---|
| 单次规划 | ✅ | `vision/src/route.py:239` `plan(map_doc, state_doc, target_spot)` → `Route`（含 `ok/nodes/distance/reason`） |
| **批量 + 计时** | ❌ 无 | 没有 benchmark 脚本 |
| `planning_ms` / `p95_ms` | ❌ 无 | 同上 |

### 最小方案

`tools/bench/route_bench.py`
```
输入: parking_map.json + 合成的 state_doc（把车放到 N 个起点）+ 全部 48 个目标
做法: 纯 Python 调用 route.plan()，用 time.perf_counter() 计时，**不经过 HTTP**
输出: 每对(起点,目标)一行: success / route_length / planning_ms
      汇总: success rate / planning_ms mean、p50、p95、max
建议矩阵: 8 个起点 × 48 个目标 = 384 条
```
**注意**：`plan()` 读的是文档快照，所以这个 benchmark **可完全离线复现**，不受相机/网络影响。

---

## E. Lock / Web / LED 时序

### 现状

❌ **四个时间戳一个都没有**（grep `user_click` / `lock_commit` / `web_update` / `led_update` → 无匹配）。

现有可替代的弱证据：
- 网页每次请求带 `?t=Date.now()`（`navigation.html`），但**不落地**
- ESP32 侧日志有 `uptime=N s`，**没有**"LED 更新时刻"
- `parking_state.json` 有 `timestamp`（整秒），可作**web_update_ts 的近似**

### 最小方案（明确标注"需要改 production 才能拿到"）

| 时间戳 | 最小实现 | 是否需改 production |
|---|---|---|
| `user_click_ts` | 网页点击时 `performance.now()`，随请求发给 `serve.py`，写一行日志 | ⚠️ **需要**（改 `serve.py` + 页面各一处） |
| `lock_commit_ts` | `serve.py` 写 `route.json` 前后各打一行时间戳 | ⚠️ 需要 |
| `web_update_ts` | 已有：`parking_state.json.timestamp`（整秒） | ✅ 不需要 |
| `esp32_led_update_ts` | 板子收到状态文档后打一行日志（`led.c` 里已有 `ESP_LOGI` 点，可复用） | ⚠️ 需要（但只是**加日志**，不改行为） |

**本次审计不动这些**。论文若要用 E 类数据，需先单独批准"允许加日志"。

---

## F. 汇总

### F1. 当前**可引用**的实测数据

| 数据 | 值 | 来源 |
|---|---|---|
| vision loop FPS（中位 / p95） | **13.8 / 38.5** | `run/vision.log`，n=39704 |
| PC source FPS（中位 / p95） | **17.7 / 27.7** | 同上 |
| HOG Hero 检测 recall / precision | **0.984 / 1.000** | `tools/hero_lab/BENCHMARK_HOG.md`（124 正 / 1240 负，三轮交叉验证） |
| HOG 单帧检测耗时 | **4.8 ms avg / 5.4 ms p95 / 209 FPS** | 同上（离线 PoC） |
| HOG 框 + 箭头定向的朝向稳定度 | **±1°（6 帧）** | 同上 |
| 48 车位状态文档可读性 | 48/48，含 `changed_ratio` | `parking_state.json` |

### F2. **缺失**的数据

- Hero：**TP/FN/FP（production 在线）**、**score/margin 落地**、**单帧处理耗时**
- 视觉：**ground-truth 定位误差**（px / cm / 度）
- occupancy：**confusion matrix、accuracy、false occupied/empty、unknown rate**
- navigation：**批量成功率、planning_ms / p95**
- 时序：**四个时间戳全部缺失**

### F3. 每项的最小补测方案（都不改 production）

| 缺失项 | 脚本 | 需要人工 |
|---|---|---|
| Hero 在线 TP/FN/FP + score + ms | `tools/bench/hero_bench.py` | 每帧标 visible（或用 B 的 gt 推导） |
| 定位误差 | `tools/bench/pose_error.py` | 60 个 ground-truth 点 |
| occupancy 混淆矩阵 | `tools/bench/occupancy_bench.py` | 每次 48 个车位的真值 |
| 路径规划批量 | `tools/bench/route_bench.py` | 无（离线） |
| 时序 E | 需**先批准**加日志 | — |

### F4. 推荐论文表格格式

**表 1 Hero 检测（离线，可按批次）**

| 验证集 | 正样本 | 负样本 | Recall | Precision | 漏检 | 误检 |
|---|---|---|---|---|---|---|

**表 2 Hero 检测（在线，需补测）**

| 视频段 | 帧数 | 检测率 | TP | FN | FP | 耗时 mean/p95 (ms) |
|---|---|---|---|---|---|---|

**表 3 视觉定位精度**

| 区域 | 样本数 | 位置误差 mean/median/p95/max (px) | 位置误差 (cm) | 朝向误差 mean/p95/max (°) |
|---|---|---|---|---|

**表 4 车位占用混淆矩阵（48 车位）**

| | 系统 EMPTY | 系统 OCCUPIED | 系统 UNKNOWN | 合计 |
|---|---|---|---|---|
| **真值 空** | | | | |
| **真值 占用** | | | | |

附：accuracy、false-occupied rate、unknown rate。

**表 5 路径规划**

| 起点数 | 目标数 | 案例数 | 成功率 | 长度 mean (px) | 规划耗时 mean / p50 / p95 / max (ms) |
|---|---|---|---|---|---|

**表 6 端到端时序（需补测）**

| 环节 | 时间戳 | 延迟 mean / p95 (ms) |
|---|---|---|
| 用户点击 → 锁定提交 | | |
| 锁定提交 → 网页更新 | | |
| 网页更新 → ESP32 灯带更新 | | |

### F5. 本次审计**未修改**任何 production 行为

- 未改 `vision/src/*`（除本文档）
- 未改 `vehicle.py` / `hero_hog.py` / `red_arrow.py` / 导航 / API / 固件
- 未调任何参数，未新增队列，未重构

---

## 附：表 5 的实测结果（2026-10-20，`tools/bench/route_bench.py`）

离线调用 `route.plan()`，不经过 HTTP、不经过相机 —— **可完全复现**。

```
命令      python tools/bench/route_bench.py --starts 20
输入      vision/out/parking_map.json（48 车位 + 路网）
         合成的 state_doc：车放在 20 个均分的路网节点上，48 个车位全部 EMPTY
案例      20 起点 × 48 目标 = 960
```

| 起点数 | 目标数 | 案例数 | 成功率 | 路径长度 mean / min / max (px) | planning ms mean / p50 / p95 / max |
|---|---|---|---|---|---|
| 8 | 48 | 384 | 100.0% | 518 / 30 / 1042 | 0.24 / 0.23 / 0.28 / 0.70 |
| **20** | **48** | **960** | **100.0%** | **510 / 29 / 1042** | **0.24 / 0.23 / 0.36 / 0.64** |

重复三次（同 960 例，看计时稳定性）：mean **0.31 / 0.24 / 0.24**，p50 **0.28 / 0.23 / 0.23**，
p95 **0.52 / 0.31 / 0.31** ms —— 第一次偏高的原因是 Python 进程首次调用时的冷启动。

**结论（只陈述事实）**：在这张地图上，**任何起点到任何车位都存在路径**（960/960），
单次规划 **亚毫秒级**（p50 0.23 ms，p95 ≤ 0.52 ms），**比相机链路的 5–100 ms 小两个数量级**，
因此在端到端延迟里可以视为可忽略项。

**这个数字的边界**：路网是**人工画的**（节点 48 个车位 + 若干道路节点），所以 100% 成功率
反映的是**这张地图的连通性**，不是"算法在任意地图上都 100%"。论文里应同时给出地图规模
（节点数 / 边数）作为上下文。

复现产物：`tools/bench/out/route_bench.csv`（每例一行：起点、目标、成功、节点数、长度、耗时、原因）

---

## 附：表 4 的实测结果（2026-10-20，`tools/bench/occupancy_bench.py`）

一帧，48 个车位全部有判断，阈值取生产的 **0.30**（从 `OccupancyParams` 读，脚本里没有第二份）。
真值由人工看图给出，**给的时候不知道系统判了什么**（因此是独立真值 —— 证据见下面的 B5）。

| | 系统 EMPTY | 系统 OCCUPIED | 系统 UNKNOWN | 合计 |
|---|---|---|---|---|
| **真值 空** | **41** | **0** | 0 | 41 |
| **真值 占用** | **1** | **6** | 0 | 7 |

```
accuracy（在有判断的 48 个车位上）  0.979
false occupied（真值空、系统报占用） 0   ← 最危险的一类，本例为 0
false empty（真值占用、系统报空）    1   ← B5
unknown rate                        0.000
```

**那 1 个 false empty 是 B5，比例为 0.298，阈值 0.30 —— 差 0.002。**
这是"刀刃上的样本"，不是算法坏了：同一位置在前一次抓帧里读到 0.61（车/光有微小变化），
说明这个车位的**可分辨度本身就在阈值附近**。论文里应把它作为**阈值敏感性**的证据保留，
而不是隐藏它。

**这份数据能说明什么、不能说明什么**
- 能说明：单帧下 false occupied 为 0，accuracy 0.979，且 B5 的失败原因是**临界值**而非误判类型
- 不能说明：**只有一帧、一种布置**。要给出稳定性结论需 ≥5 次（全空/单车/多车/压线/光照变化）
