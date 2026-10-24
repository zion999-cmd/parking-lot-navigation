# HERO 识别链路：四层分工与实测

本文描述**当前**的 HERO（英雄车）识别链路，以及支撑每一个决定的测量。任何一条数字都可以用
`tools/` 下的工具重新跑出来；跑不出来的结论不写在这里。

## 0. 四层分工

```
Camera 800x600
   ↓
① HSV 召回            vision/src/hsv_marker.py      颜色 → 红色候选（宽松，不判身份）
   ↓
② 形状证据            40 个真实采样的 reference contour，取 matchShapes 最小值
   ↓
③ 身份判断            vision/src/hero_belief.py     position / motion / size / colour / shape 综合
   ↓
④ 状态机与连续性        ACQUIRED / TRACKING / UNCERTAIN / LOST / REACQUIRING
   ↓
朝向                  vision/src/red_arrow.py      4x 上采样 ROI → PCA + 头尾
   ↓
30° 门 + 12 帧持续     vision/src/heading_state.py
```

一句话：**HSV 负责召回，bank 负责形状证据，belief 负责身份，状态机负责连续性。**
四层任何一层被要求做别人的事，都会退化——这一点是实测出来的，不是设计偏好。

## 1. 颜色：箭头不是"红"

箭头在 HSV 里是 **H 24–46（中位 31，OrangeRed）**，不在教科书红色区间 0–10/170–180 里。
按红色区间取 mask 得到的是画面里另一个红物（H≈11），并且它**评分合格、位置稳定**——干净而
错误。实测：254 个候选 100% 落在同一格，箭头一个都没有。

生产用 **H 0–50 / S≥60 / V≥40**（`hsv_marker`）。

## 2. 召回：多阈值，因为一个阈值服务不了全桌

```
亮区 (240,485)  S≈115   S≥60 命中 105/120
暗角 (68,475)   S≈55    S≥60 只留下饱和核心 → 18x30 菱形 → shape 0.91 被拒 12/12
                         S≥45 恢复箭头 32x17 shape 0.13
但 S≥45 在亮区会把箭头和周围红物糊在一起，掉到 9/120
```

所以 `S ∈ {60,50,45}` 只做**候选生成器**，按中心 8 px 去重，shape 最好的那份留下。
先裁窗再阈值：局部 90 px 窗 **1.9 ms**，全图 3 阈值 **9.1 ms**。

## 3. 形状：从"门"降级为"证据"

40 个 reference（32 个操作员真实采样 + 8 个程序化旋转）启动时加载一次（250 ms，之后
0.002 ms）。每个候选取 40 个 `matchShapes` 的最小值。

32 条标定（8 位置 × 4 朝向）实测：

| 策略 | candidate recall | selection |
|---|---|---|
| shape 硬门 0.60 + 单参考（旧） | 25/32 | 20/32 |
| shape 硬门 0.60 + 40 参考 | 30/32 | 28/32 |
| **shape 只作证据 + 40 参考** | **30/32** | **29/32** |

按 30 px 参照点计（诚实口径，见下）**32/32**。

**旧策略的实际代价**：7 个位置的箭头**就在正确位置、shape 0.03–0.06**，却因为"形状门"连候选
池都进不去。

⚠️ **口径警告**：标定文件里的 `centre` 是"当时点到的连通域质心"，不是 ground truth。三条样本
的记录 blob 是被遮挡/粘连的版本（例如记录 37×36 area 1159 H28，而流程找到的是 18×32 area 285
H 0.046）。用 14 px 判据会把"找到了更好的箭头实例"算成失败。

## 4. 身份：状态相关的权重

```
ACQUIRED / TRACKING   position 0.40  size 0.20  shape 0.20  colour 0.10  area 0.10
UNCERTAIN             position 0.25  size 0.20  shape 0.30  colour 0.15  area 0.10
LOST / REACQUIRING    position 0.15  size 0.20  shape 0.40  colour 0.15  area 0.10
```

tracking 时位置是最强证据，shape 只做仲裁；一旦不确定，位置先验已经失败过，shape 与外观才
接管。**shape 权重初版保持 0.20，没有为了过测试去调它。**

## 5. 运动：能被推，不能被瞬移

```
predicted = last_position + velocity * dt        velocity: EMA α=0.45, |v| ≤ 60 px/frame
search_radius = 22 px（静止） → 最高 180 px（快速）   随速度放宽
max_prediction = 120 px
```

原则：**车可以移动，但不能在两帧之间漂过半个停车场。** 静止时窗口收紧，推车时自动放宽。
远离预测点的候选**不能单帧接管**，需连续 6 帧一致；1–3 帧的闪跳直接丢弃。

## 6. 自学习：只有确定时才能学

外观记忆（area / w / h / aspect / median H,S,V / shape）是唯一**会被污染且不可恢复**的部分：
位置信错一会儿能纠正，外观信错之后，一切比较的基准都错了。

四道闸门：

| 闸门 | 规则 |
|---|---|
| 只允许 `accept` 学 | `acquire`/`takeover` 不学；冷启动可学（无 profile 可污染）；重新获取不学 |
| 置信度下限 | `< 0.55` 不学 |
| 接管后观察期 | `takeover` 后 **20 帧**（≈2 s @10fps）不学：它会占位置，不能定义外观 |
| profile 锁定 | 累积 25 次更新后，只接受 `> 0.75` 的候选，且学习率降到 1/3 |

实测（原来会投毒的序列：accept → 远处"看起来像"的候选 ×7 → 之后 4 帧普通 accept）：
profile 保持 `(area 270, S 110, V 160)` 不变，跳过学习 9 次，位置确实移动。

**已知残留行为**：hero 被拿走后，系统仍可能在 6 帧后把桌上的另一个红物当作 HERO 并移动过去。
区别在于 profile 不再被污染，hero 放回来立刻认得出。更保守的"HERO 缺席判定"暂未实现。

## 7. 朝向：两个独立问题

| 问题 | 实测 | 解法 |
|---|---|---|
| 斜角几何余量塌陷 | 0/90/180/270° 余量 2.72 px、抖动 0.0°；45/135/225/315° 余量 **0.51 px**、抖动 **77.7°**、翻转 **114/228** | ROI **4× 上采样**后做 PCA+头尾：斜角抖动 **0.7°**、翻转 **0/228**、余量 **10.62 px**，且更快（1.87 vs 7.39 ms） |
| 单帧 180° 翻转 | 历史实测 185/5/182/4/186/5（二十帧） | `heading_state`：30° 门 + 12 帧持续。连续转动（+5°/帧）**零延迟**；瞬时跳 45° 需 12 帧；单次翻转**永不相信** |

`matchShapes` 是旋转不变的，所以**它给不出朝向**——朝向必须来自 PCA + 头尾判别。

## 8. 失败策略

```
位置：UNCERTAIN 期间保留最后可信位置并标 stale；LOST 时返回"不可见"，不静默替换成别的红块
朝向：当前帧失败 → 沿用已提交方向并标 heading_stale；从无历史 → null，不伪造
```

## 9. 已删除/降级的东西，以及为什么

| 组件 | 状态 | 依据 |
|---|---|---|
| HOG + Linear SVM | 退出运行时，代码/模型/文档保留 | 220 帧回归里命中 53/220、75.7 ms；`red_arrow` 220/220、4.5 ms |
| `detect_in_box` | 不再被调用 | 参考轮廓取到的是 400×400 白纸矩形（4 点），`matchShapes` 恒 > 0.30，**永远返回 None** |
| template matching（0–345°） | 未采用 | 800×600 下两种变体都失败：一个分不开（0.371 vs 0.296），一个退化（1.000 vs 0.994） |
| 膨胀分组（fragment grouping） | 未采用 | 28/32 箭头本就是**完整单块**（覆盖率 1.00）；膨胀 9 px 在 **31/32** 上会与邻居粘连 |

## 10. 未验证

**动态推车测试尚未执行。** 静态帧与"原地转方向"都无法考验运动门控：220 帧回放里新旧实现
逐位相同（0.000 px / 0.000°），因为那段素材箭头干净、两种实现都 100%——它证明的是**不退化**，
不是**改善**。

工具已就绪：

```bash
python tools/bench/hero_dynamic.py record --seconds 60 --label A_left_to_right
python tools/bench/hero_dynamic.py replay <frames_dir> --impl HEAD       # 旧实现
python tools/bench/hero_dynamic.py replay <frames_dir> --impl current    # 新实现
python tools/bench/hero_dynamic.py compare <old.csv> <new.csv>
```

时间线逐帧记录 `x,y,heading,confidence,state,candidate_count` 与五项证据分数
（`position/motion/size/colour/shape`），跳错时能直接看出是哪项把错误候选推上去的。

## 11. 复现工具一览

| 工具 | 用途 |
|---|---|
| `tools/hero_lab/hsv_spike/` | HSV 三阶段 spike：候选、shape、template（含结论） |
| `tools/hero_lab/angle_sweep/` | 24 角度扫描、门控序列 |
| `tools/hero_lab/heading_spike/AUDIT.md` | `red_arrow` 逐行考古 + 我自己的三处测量错误 |
| `tools/hero_lab/fragment_grouping.md` | 分组假设的否证 |
| `tools/hero_lab/shape_feature_ablation.py` | shape 门 vs shape 证据 |
| `tools/label_server.py`（`./start.sh label`） | HSV 标定模式：点箭头 → 记录颜色/几何/三阈值结果 |
| `tools/bench/hero_dynamic.py` | 动态录制 / 回放 / 新旧对比 |
| `tools/bench/check_hog_removed.py` | HOG 退出回归 |
