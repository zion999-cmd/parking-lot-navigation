# project.md —— 一页看全这个项目

**状态 / 工具层 / 使用方法**三件事都在这一页。
深入细节看 `README.md`（硬件证据链、设计取舍、踩坑全过程）和 `vision/README.md`。

当前固件版本：**0.2.11**　｜　车位标定：**48 个**（`vision/config/parking_spots.json`）

---

## 1. 项目是什么

给 11 岁小朋友的科创比赛做的「**停车场车位级导航沙盘**」。

不是一个从零发明的技术，而是一次**系统集成**：把成熟的现成技术串成一条完整的链路。

```
ESP32-S3-CAM 节点（纯视频传感器）
        │  LAN / MJPEG 流  ← 只做这一件事，不跑任何视觉算法
        ▼
PC 端 OpenCV 车位占用检测（几何方法，无深度学习）
        │  signal / 状态服务 ◄── 下一步（P0003）
        ▼
ESP32 导航小车（车位级导航）◄── 下一步
```

**为什么把视觉放在 PC 上**：ESP32 只负责稳定出图，算法迭代放 PC 上最快；相机节点固件因此可以冻结，不被算法改动牵连。

---

## 2. 状态

### 2.1 阶段

| 阶段 | 内容 | 状态 | 关键证据 |
|---|---|---|---|
| **P0001** | ESP32-S3-CAM 作为局域网 MJPEG 相机 | ✅ **已验收** | 800x600 SVGA / JPEG、`/stream` 稳定 20+ fps、`/` 状态页、启动日志四条约定齐全 |
| **P0001.1** | Wi-Fi 配网（SoftAP+强制门户+NVS）＋ 薄 OTA（A/B 分区、SHA256、原生回滚） | ✅ **已验收** | **0.2.10 → 0.2.11 真机 OTA 升级成功，20 秒，`slot ota_0` → `slot ota_1`，全程不按 BOOT 键** |
| **P0001.2** | 板载 TFT 状态屏（ST7789 240x240） | ⚠️ **代码就绪，画面未目视确认** | 分带刷新修复后不再 OOM；`disp=` 计数正常递增；**但从没人看过屏幕到底显示了什么** |
| **P0002** | PC 端车位占用检测（OpenCV）＋ **停车场地图** ＋ **输出契约** | ✅ **已可用** | 48 个车位空/占用识别正确（静止场景下 48/48 稳定为 0 差异）；地图：**48/48 车位入口自动派生、全部从入口可达**，两端车位可手工纠正；契约：`parking_map.json` + `parking_state.json`（含 `UNKNOWN`）—— 见 §7 |

### 2.2 实测数字（固件 0.2.9~0.2.11）

| 指标 | 数值 |
|---|---|
| 视频流帧率 | **28.3 fps**（设备侧自报）／ 13.7~15 fps（PC 端 Python 客户端，受反压影响） |
| 带宽 | **464~511 KB/s**（2.3~4.1 Mbps），单帧 14~20 KB |
| 单帧耗时拆分 | **grab ≈ 20 ms**（等传感器）／ **send ≈ 15.5 ms**（写 socket） |
| 发送失败 | **0** |
| 网络 | RSSI −30~−42 dBm，ping 丢包 0%，发射功率 10 dBm |
| 内存 | `heap_internal` ~123 KB（streaming）／~129 KB（idle），**无泄漏** |
| 视觉阈值 | `changed_ratio ≥ 0.15` 判占用，逐像素阈值 25（灰阶），blur 5 |
| 车位几何 | 48 个，全部 47~48 × 100~104 px，**两两重叠 0 对** |
| 测试 | **135 项全绿**（occupancy 17 / detector 17 / editor 37 / alignment 15 / road 49） |

### 2.3 还没验证的（诚实清单）

- **P0001.2 的屏幕画面没人看过** —— 需要有人盯一眼 LCD，确认显示的是 IP 等状态信息
- **OTA 回滚（SC-16）未实测** —— A/B 双槽 + IDF 原生回滚 + `MarkCurrentVersionValid` 都已就位，但没故意制造一次失败启动去验证
- **视觉的阈值还没有用真实案例拟合** —— `vision/data/cases/` **目录还不存在**，也就是 5 个验收案例一次都没录过；阈值 0.15 目前是经验值，不是测量值（`tests/fit_threshold.py` 就是干这个的）
- **相机被物理碰歪后的自动对齐无效** —— 见 §6

---

## 3. 硬件

| | |
|---|---|
| 模组 | ESP32-S3（QFN56）rev v0.2，MAC `14:c1:9f:28:a1:c8` |
| 内存 | 8 MB OCT PSRAM @80 MHz ＋ 16 MB flash（Boya） |
| 相机 | **OV3660**，SCCB 地址 0x3c，DVP 并口（GPIO 4~18） |
| 屏幕 | ST7789 240x240，SPI3 @40 MHz（CS45 / DC47 / RST21 / SCLK19 / MOSI20 / BL38） |
| 串口 | USB-UART **CH340**，当前 `/dev/cu.usbserial-14320`（会变，用 `ls /dev/cu.*` 确认） |
| 网络 | `HUAWEI-1B9C59` 2.4 GHz ch6，设备 IP **192.168.3.118** |

### ⚠️ 这块板子**没有自动下载电路**

**首次/手动烧录必须让芯片进下载模式**：按住 **BOOT**，插 USB（或按 RESET），等 `idf.py flash` 打出 `Connecting...` 再松手。

> 但**日常升级不用烧录** —— OTA 已经打通，见 §5.2。

---

## 4. 工具层

### 4.1 固件（`firmware/esp32-s3-cam/`）

| 文件 | 作用 |
|---|---|
| `main/main.c` | 启动顺序、网络状态机、**诊断行（每 10 秒一行）** |
| `main/camera.c` | OV3660 初始化（引脚表附完整证据链） |
| `main/stream.c` | **MJPEG 流服务** + 状态页 + 手动 OTA 触发；含逐帧 `grab/send` 计时 |
| `main/wifi.cpp` | 复用 `78/esp-wifi-connect` 的 C 适配层；省电/发射功率强制与读回 |
| `main/provisioning.cpp` | 配网状态机（SoftAP ⇄ STA） |
| `main/ota.c` | 薄 OTA：拉 manifest → 比对版本 → 下载 → SHA256 → 写非活动槽 |
| `main/display.c` | TFT 状态屏，分带刷新（16 行/带，7.5 KB bounce buffer） |
| `main/Kconfig.projbuild` | **三个可调项**（见下） |

**Kconfig 可调项**（`idf.py menuconfig` → Parking Lot Camera）：

| 选项 | 默认 | 说明 |
|---|---|---|
| `CONFIG_OTA_MANIFEST_URL` | `http://192.168.3.102:8000/manifest.json` | **全工程唯一的 OTA 地址来源**，改了要重新编译并烧录 |
| `CONFIG_WIFI_MAX_TX_POWER_DBM` | **10** | 发射功率上限。**故意不是 20** —— 板子 3.3V 轨供不动 20 dBm 的电流尖峰，见 §6 |
| `CONFIG_DISPLAY_ENABLE` | y | 不装屏幕时关掉 |

### 4.2 OTA 服务（`tools/ota-server/`）

| 脚本 | 用法 |
|---|---|
| `./run.sh` | 启动（默认 8000 端口、所有网卡），启动时会打印本机 IP 和设备的正确 URL |
| `./publish.sh <版本>` | 发布：从 `build/parking-lot-cam.bin` 读版本 → 复制 → 算 SHA256 → 写 manifest |
| `./publish.sh <版本> <bin>` | 发布指定文件 |
| `./publish.sh --sha256 <64位十六进制> <版本>` | **故意写错摘要**，用来验证设备会拒绝 |
| `./publish.sh --allow-version-mismatch ...` | 强制允许版本号与内嵌版本不一致 |

**保护机制**：`publish.sh` 会读 bin 里偏移 `0x30` 的内嵌版本号，不一致就拒绝发布。

### 4.3 视觉工具（`vision/`）

```zsh
P=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
```

| 工具 | 作用 | 常用命令 |
|---|---|---|
| `src/main.py` | **主程序**：实时画面 + 占用判定 + 图标 | `$P src/main.py` |
| `src/calibration.py` | 标定/编辑车位框（Auto Detect 是助手，人工确认才是准） | `$P src/calibration.py` |
| `src/spot_editor.py` | 编辑模型（状态与操作），被上面两个复用 | — |
| `src/spot_config.py` | 车位配置读写（手改友好格式、原子写） | — |
| `src/occupancy.py` | 占用算法（参考帧 + absdiff + 阈值 + 形态学） | — |
| `src/alignment.py` | 相机漂移补偿（ECC + 结构相关度门控） | — |
| `src/detector.py` | 车位几何自动检测（Profile 法，非深度学习） | — |
| `src/camera.py` | MJPEG 读取（`cv2.VideoCapture` 拿不到这个流，所以自己实现） | — |
| `src/map_document.py` | **地图文档读写**（一个文件两个段落，互不覆盖、原子写） | — |
| `src/road_network.py` | **道路数据模型** + `derive()`（车位+道路 → 导航图，纯函数） | — |
| `src/road_editor.py` | **画道路**的编辑模型（沿路点击、拖点、交汇） | — |
| `src/contract.py` | **P0002 输出契约**：生成 `parking_map` / `parking_state` 两个文档（§7） | — |
| `src/emit_map.py` | **发射静态契约**：配置 → `out/parking_map.json` | `$P src/emit_map.py [--strict]` |
| `tests/demo_alignment.py` | 对齐演示 / 测当前偏移 | `$P tests/demo_alignment.py --shift 0,0 --frame X.jpg` |
| `tests/fit_threshold.py` | **从验收案例拟合阈值**（阈值要测量，不要猜） | `$P tests/fit_threshold.py --cases 'data/cases/*.csv'` |

### 4.4 测试

```zsh
tools/tests/run.sh                       # OTA 版本比较规则（宿主测试，纯 C）
cd vision && for t in tests/test_*.py; do $P $t; done    # 135 项
```

---

## 5. 使用方法

### 5.0 环境（每次开新终端）

```zsh
conda activate esp32                    # ⚠️ 必须在 export.sh 之前
. ~/Esp32/esp-idf/export.sh
```

> 先 conda 再 export 是硬要求：否则 IDF 会去找并不存在的 `idf5.5_py3.14_env`。

### 5.1 编译 / 手动烧录

```zsh
cd firmware/esp32-s3-cam
idf.py build
idf.py -p /dev/cu.usbserial-14320 flash     # 按住 BOOT 到 "Chip is ESP32-S3"
idf.py -p /dev/cu.usbserial-14320 monitor   # 看日志（Ctrl+] 退出）
```

启动日志必须出现四条（P0001 约定）：

```
WiFi connected
IP address: 192.168.3.118
Camera initialized
Stream server started
```

### 5.2 日常升级：**不用再按 BOOT 键**

```zsh
# ① 改代码，改版本号（firmware/esp32-s3-cam/CMakeLists.txt 的 PROJECT_VER）
# ② 编译
cd firmware/esp32-s3-cam && idf.py build
# ③ 发布到 OTA 服务器
cd ../../tools/ota-server && ./run.sh &        # 没在跑的话
./publish.sh 0.2.12
# ④ 让设备自己去拉
curl -sS -X POST http://192.168.3.118/api/ota/check
```

**约 20 秒后设备自动重启到新版本**，`/` 状态页会显示新的版本号和新槽位（`ota_0` ⇄ `ota_1` 交替）。

设备端日志应该出现：

```
ota: checking http://.../manifest.json (running 0.2.11)
ota: GET http://.../manifest.json -> HTTP 200, content length ...
ota: downloaded ... bytes
```

### 5.3 视觉：标定 → 抓参考帧 → 看占用

**这是唯一正确的操作顺序**（顺序错了就会出现「空车位也报占用」，见 §6）：

```zsh
cd vision
P=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
$P src/main.py
```

| 步骤 | 操作 |
|---|---|
| **① 对齐车位框** | 窗口里按 **`e`** 进编辑模式 → `m` 切移动范围（当前格 / **整行** / 全部）→ `2/4/6/8` 微调 → `s` 保存 → `q` 退出 |
| **② 别碰相机** | 之后相机保持不动 |
| **③ 清空所有车位，按 `r`** | **抓参考帧必须是最后一步** |
| **④ 放车 / 移走** | 对应车位应立即 EMPTY ⇄ OCCUPIED |
| **⑤ 画道路**（导航用） | 编辑模式里按 **`t`** 或点顶部 **[ Roads ]** → **沿车能走的路连续点击**（拖点可移动、右键点已有路点可交汇、`n` 另起一条）→ 按 `p` 看检查结果 → `s` 保存 |
| **⑥ （只在需要时）纠正车位入口** | `Roads` 层里**把橙色的入口小点拖到车实际进来的那条边**上 —— 一排两端的车位可能需要；改完自动生效，不用别的操作 |

> **画完道路就结束了** —— 每个车位的入口是系统自动派生的（见 §6），不需要任何绑定步骤。
> 按 **`p`** 看检查结果。

**`main.py` 按键**：

| 键 | 作用 |
|---|---|
| `r` | **抓参考帧**（必须是全空状态） |
| `e` | 进编辑模式（就地改车位框） |
| `t` | **参考帧体检**：区分「画面整体变了」和「这几格内容真的不同」 |
| `p` | 打印逐格数字（`*` = 超阈值） |
| `g` / `d` | 显示内框 / 差异叠加 |
| `+` `-` | 阈值 ±0.01　｜　`[` `]` 逐像素阈值 ±5 |
| `s` | 存一张标注截图到 `data/debug/` |
| `q` / `ESC` | 退出 |

**编辑模式（`e`）顶部有两个页签：`[ Parking Spots ] [ Roads ]`**（按 `t` 切换，或直接点）。

**Parking Spots 层**（摄像头的问题）：左键拖角/拖体、右键吸附到边、`2/4/6/8` 微调 5px、`Tab` 切换、`m` 切移动范围、`n` 增、`d` 删、`g`/`G` 复制整行、`a` 自动检测、`i` 内框比例、`A` 对齐、`z` 撤销、`c` 清空、`s` 保存。
（`r` 在这一层仍是「抓参考帧」；改名用 **`R`**。）

**Road Road 层**（车能开哪里，**纯手工，不做自动检测**）：

| 操作 | 键/鼠标 |
|---|---|
| **画路** | **沿路连续点击**（下一个点自动接上一个点，连成一条路） |
| 调整 | 拖动点，或 `2/4/6/8` 移动选中的点 |
| 两条路交汇 | **右键点一个已有的路点** |
| 另起一条路 | `n`（或右键点空白） |
| 标记入口 | `k`（可选：默认自动取最靠近画面边缘的路点） |
| 删除点 | `d`（连同它的路段） |
| **纠正某个车位的入口** | **把橙色入口小点拖到车实际进来的那条边**（吸附到最近的边；指定过的会多一圈洋红色环） |
| 检查地图 | `p`（地图语言的结论：哪些车位离路太远 / 无法到达） |
| 其他 | `z` 撤销　`c` 清空　`s` 保存　`Tab` 下一个点 |

**没有「绑定车位」这类按键** —— 车位接到道路是**派生数据**，系统自动完成（§6）。
**车位入口改回自动**：回到 `Parking Spots` 层选中那个车位按 **`j`**。

**录验收案例**（`data/cases/` 目录目前还不存在，第一次用得自己建）：

```zsh
mkdir -p data/cases
$P src/main.py --headless --duration 20 --csv data/cases/case1_all_empty.csv
# 依次录 5 个：全空 / 单格有车 / 两格有车 / 全满 / 移走后再录
$P tests/fit_threshold.py --cases 'data/cases/*.csv' --write-template data/cases/labels.json
# 然后手工编辑 labels.json 标注真值，再跑一次 → 得到测量出来的阈值和逐格余量
```

### 5.4 设备诊断行（每 10 秒一行，排障看这个）

```
uptime=240s fw=0.2.11 fps=28.3 frames=842 clients=1 rssi=-33 dBm net=CONNECTED
ip=192.168.3.118 ps=none tx=10.00dBm link="ch6+0 11n/HT20 -33 dBm" disp=6
heap_internal=123211 dma_free=115423 heap_psram=7775328 heap_min=7843980
```

| 字段 | 该看什么 |
|---|---|
| `ps=` | **必须是 `none`**。是 `min_modem`/`max_modem` → 射频在睡，流会慢十倍 |
| `tx=` | 应是 `10.00dBm`。被别的值占用了说明组件从 NVS 恢复了旧设置 |
| `link=` | 若是 `802.11b` → 链路被限在 1~11 Mbps |
| `disp=` | 面板刷屏次数。**帧率掉的时候它不动 = 屏清白了** |
| `dma_free=` | 内部 DMA 内存。与 SPI bounce buffer、Wi-Fi 发射抢同一块 |
| `frames=` | **真实帧数增量**。别信 `fps=`（瞬时 EMA，卡顿后会飙到 200+） |

流任务还有一行按帧计时的：

```
stream: 50 frames in 1.8 s (28.3 fps) | grab avg 19.9 ms (max 30.6) |
        send avg 15.5 ms (max 94.8) | jpeg avg 16.4 KB | 464 KB/s | send failures 0
```

**`grab`（等摄像头）vs `send`（写 socket）**——哪个大就是哪边的锅。这一行是定位帧率问题的转折点。

### 5.5 跑测试

```zsh
tools/tests/run.sh
cd vision && for t in tests/test_*.py; do $P $t; done
```

---

## 6. 停车场地图：车位 + 道路

用户只需要做两件事，第三、四件是系统做的：

| | 谁做 | 做什么 |
|---|---|---|
| ① 标停车位 | 人 | 现在的车位标定（48 个 polygon） |
| ② 画道路 | 人 | **只画车能走的路**（沿路连续点击，可拖点、可交汇） |
| ③ 把车位接到道路 | **系统** | 派生数据：不用编辑、不用按任何键 |
| ④ 转成导航 Graph | **系统** | `derive()`，纯函数，要用时现算 |

**车位的入口是推导出来的**：车位**临路的那条边的中点**。如果它落在一条路的中间，
**系统会在那条路里插一个点** —— 不要求用户创建任何节点，也不要求用户理解
node / edge / chain / graph。那些只是数据的形状，不是任务的一部分。

**为什么这是推导而不是检测**：「车位朝哪边开口」由两个**人的决定**推出
（车位多边形 + 你画的路），**完全不看画面**。道路本身也不做自动识别 ——
车能走哪里是人的判断。

**屏幕上分得很清楚**：

| 画法 | 含义 |
|---|---|
| **实线 + 大圆点**（绿 = 入口，灰 = 路点） | **你画的路** |
| **细虚线 + 橙色小点** | **系统派生的车位入口** |

### 用户看到的就是这些

```
ROADS   7 points, 7 segments   |   48/48 bay entrances derived
```

按 `p` 得到的是地图语言的检查结果，不是图论报告：

```
✅ map complete: 48 bay entrances derived, all reachable from the entrance
❌ 3 bay(s) are a long way from any road: 27 (309px), 28 (289px), 26 (270px)
   （示例：这三个车位前面没路，最近的也在 300px 外 —— 补一条通道就好）
❌ 2 bay(s) cannot be reached from the entrance: 38, 39
```

**入口也不用用户指定**：默认取**最靠近画面边缘的那个路点**（车从外面进来）。猜错了可以选中那个点按 `k` 覆盖。

### 文件里存什么

**只存人的两个决定**：

```json
{
  "image_size": [600, 800],
  "spots": [ { "id": "38", "polygon": [[...]], "entrance": 2 } ],   ← ① 人标的车位
                                                              （entrance = 人指定从第几条边进，
                                                                不写就是「朝向最近的路」）
  "nodes": [ { "id": "n1", "x": 77, "y": 496 } ],               ← ② 人画的道路
  "edges": [ { "a": "n1", "b": "n2" } ]
}
```

**派生数据不存**（`derive()` 是纯函数，不修改原数据），因此：

- 反复调用**不会累积**支线
- 文件里永远是「人做的决定」，不会混进机器生成的东西
- 车位挪了**不会留下过期的连接**
- `nodes` 里**不会出现** `kind: "spot"`（有测试守着）

**两层互不破坏是代码保证的**：每层保存时会把它读到的所有段落原样重写
（`map_document.build()`）—— 写道路不会丢车位，写车位不会丢道路。

### 车位入口可以手工纠正（尾排两端、拐角的情况）

**自动规则在哪里不够用**：车位的入口 = **朝向最近那段路的那条边**。绝大多数车位都对，
但在一排的**两端**，「本排旁边的路」和「外环侧边」的距离势均力敌 ——
**这不是测量能定的，是判断**，而看图的人才能做这个判断。

实测这张图：**37、38**（第二、三排最左）最近的是外环**竖直段** → 开口落在**左**面 ✗；
**25、50**（最右）→ 落在**右**面 ✗。它们所在的那一排前面本来就没有路。

**怎么改**：在 `Roads` 层，**用鼠标把那个橙色入口小点拖到「车实际进来的那条边」上**，
松手时吸附到该车位最近的那条边（按**到边的距离**算，长边更容易点中）。

**改完不需要任何其他动作**：接入点自动移到那条边、道路自动在落点插一个点、图和路径全部重建。
这就是「派生」的好处 —— 人只需改**一个判断**，其余自动跟随。

| 画面上 | 含义 |
|---|---|
| 橙色小点 | 系统**推导**出来的入口 |
| **多一圈洋红色环** | **人手工指定过边**的入口 |

表头会数给你看：`48/48 bay entrances derived, 4 put there by hand`

**改回自动**：在 `Parking Spots` 层选中那个车位按 **`j`**（`z` 可撤销）。

> **另一种修法（结构性）**：在中排车位前面补一条横通道 —— 那一排的「最近路」就变成通道，
> 开口会**自动**变对，不必手工拖。两者任选，补通道更贴近车实际走的路线。

### 导航侧接口（P0003）

```python
net   = RoadNetwork.load("config/parking_spots.json")   # 人画的道路
graph = net.derive(spots)                               # 一行：道路 + 每个车位的入口
graph.adjacency()      # 带权邻接表（单向边只出现一次）→ A* / Dijkstra 直接可用
graph.spot_nodes()     # {"38": "n9"}   车位号 → 图上的节点
```

### 道路画在哪，决定最后那一小段

车位到路的距离 = 车要开的最后一段。**这张图实测（外环 + 4 个手工纠正的入口）**：

| | 入口引线 |
|---|---|
| 最短 | **27px** |
| 中位 | **32px** |
| 最长 | **38px** |
| 离路 >220px 的车位 | **0 个** |

也就是说**外环路已经够用**，不需要为了缩短引线去补通道。

换个画法也不必担心：引线长度是由「车位的开口到最近那段路」决定的，接入点是从**开口**（而不是车位中心）找最近的路 —— 所以把入口改到正确的那条边，本身就会让最后一段变短。实测把 37/38/25/50 从侧面改到上/下边后，最长的引线从 53px 降到 38px。

> 什么时候真需要补通道：`p` 报告「N 个车位离路很远」时（那几个车位前面确实没路）。
> 通道不需要精确，画在那些车位前面即可。

> 你现在的 7 点外环路 + 48 个车位，检查结果已经是 `地图完整`（48/48 派生、48/48 可达）。
> 补通道只是让路径更贴近车实际走的路线。

## 7. P0002 输出契约

**这是最重要的一刀。** 契约一旦固定，P0002 就封版：后面只认这两个文档，
**完全不再关心摄像头、OpenCV、ROI、changed_ratio 是怎么来的。**

P0002 只回答两类问题：

> **① 停车场长什么样？** —— 静态，`parking_map.json`
> **② 现在停车场发生了什么？** —— 实时，`parking_state.json`

### 7.1 静态世界：`parking_map.json`

```zsh
$P src/emit_map.py                 # config/parking_spots.json → out/parking_map.json
$P src/emit_map.py --strict        # 有任何车位没接入道路就返回非零（可用作检查）
```

```json
{
  "schema": "parking_map/1",
  "coordinate_system": { "type": "image_2d", "width": 800, "height": 600, "origin": "top_left" },
  "spots": [
    { "id": "38",
      "polygon": [[64, 296], [112, 296], [112, 400], [64, 400]],
      "access": { "point": [88, 400], "road_node": "r12" } }
  ],
  "road_network": {
    "nodes": [ { "id": "r1", "x": 77, "y": 496, "type": "entrance" } ],
    "edges": [ { "from": "r1", "to": "r2", "distance": 58, "bidirectional": true } ]
  },
  "entrances": [ { "id": "main_gate", "node": "r1", "type": "entry_exit" } ]
}
```

| 字段 | 说明 |
|---|---|
| `coordinate_system` | **全项目唯一坐标系**：像素、原点左上、y 向下。车位／道路／主车／路径全在这里 |
| `spots[].access` | **车位接入点**（车从道路进入这个车位的位置）＋ 它连的道路节点 |
| `spots[].access` **≠** `entrances` | `access` = 车位入口；`entrances` = **停车场大门**。两者绝不能混 |
| `road_network` | **已经推导好的导航图** —— P0003 **不要**自己从道路中心线重新推导 |
| 节点 id | `r1, r2, ...` |
| `edges[].distance` / `bidirectional` | 权重（像素距离）与方向；单向路为 `bidirectional: false` |

**节点/边是内部表达**：用户在地图工具里只做一件事 —— **画车能走的路**，其余 P0002 自动生成。

### 7.2 实时世界：`parking_state.json`

`main.py` 运行时写出（默认 `out/parking_state.json`，0.5 秒一次；`--state-out` / `--state-every` 可调）：

```json
{
  "schema": "parking_state/1",
  "timestamp": 1791283027,
  "spots": {
    "38": { "state": "OCCUPIED", "changed_ratio": 0.42 },
    "39": { "state": "EMPTY" },
    "40": { "state": "UNKNOWN", "reason": "no frames for 4.0s" }
  },
  "vehicle": { "id": "hero", "x": 423, "y": 286, "heading": 91.4, "visible": true }
}
```

**状态只有三种，`UNKNOWN` 是必须的：**

| 状态 | 何时 |
|---|---|
| `EMPTY` | 与空置参考帧一致 |
| `OCCUPIED` | 差异超过阈值 |
| **`UNKNOWN`** | **没有参考帧 ／ 相机或网络停了 ／ 车位被遮挡 ／ 图像异常 ／ 置信度不足** |

**为什么 `UNKNOWN` 不能省**：导航可以被告知「回来再看一次」，**但无法从「明明不知道却说是空的」里恢复**。
所以「没有参考帧」「超过 2 秒没有新帧」会让**全部车位**变成 `UNKNOWN` 并带上 `reason`，
**而不是假装 EMPTY**。程序因此即使在启动失败时也会先写出一份合法的状态文档。

`changed_ratio` **只是诊断信息**：消费方可以显示它，**不能用它做决策**（阈值是 P0002 的内部事务）。

### 7.3 P0002 说事实，不说决策

| P0002 **可以**说 | P0002 **不该**说 |
|---|---|
| 38 是空的 | 38 是**最适合**的车位 |
| 38 的接入点在 (88,400)，连道路节点 r12 | 应该走 r12 → r17 → 38 |
| 主车现在 (423,286)，朝向 91° | 主车应该向右转 |

前三个是**事实**，后三个是**决策与规划** —— 属于 P0003。

### 7.4 当前完成度

**已完成**：48 个车位 polygon　车位 `access`　道路 ＋ 节点/边　停车场大门　车位→道路连接
　EMPTY / OCCUPIED　**`UNKNOWN` 接口**　两个契约文档

**还差两项**：
1. **最终阈值实测确定**（现在 0.15 仍是经验值，见 §2.3）
2. **主车 `x/y/heading`** —— P0002.5，见 §10

### 7.5 三个文件的分工（别混）

| 文件 | 角色 | 谁写 |
|---|---|---|
| `vision/config/parking_spots.json` | **人画的**：车位 ＋ 道路（只含决定，可手工编辑） | 地图工具 |
| `vision/out/parking_map.json` | **生成的契约**：含派生出的接入点与导航图 | `src/emit_map.py` |
| `vision/out/parking_state.json` | **实时契约**：状态 ＋ 主车 | `src/main.py` |

---

## 8. 已知限制与坑

### `align` 默认关闭：它的语义在这里不成立

窗口顶部会显示 `align off`，或者 `align ignored: the scene itself differs...`。**这是正常的，不是故障。**

实测（同一台相机、同一套标定）：

| 比较 | 结构相关度 |
|---|---|
| 实时帧 vs 实时帧（相隔几秒） | **0.996** ← 相机纹丝不动 |
| 参考帧 vs 实时帧 | **0.613** ← 低于 0.80 门槛 |
| 对齐器以参考帧为基准 | 拒绝 |
| 对齐器以另一张实时帧为基准 | 接受（0.996） |

原因：**占用检测要的是「空车位状态」，而对齐要的是「场景结构锚点」，两者不兼容** —— 停了车以后画面结构就变了（参考帧与实时帧的差异像素 62% 落在车位内），所以以参考帧为基准的对齐必然失败。

结论：**固定相机不需要对齐**（0.996 说明没有漂移），所以 `--align` 默认关闭；相机真被碰歪了就用 `e` 手动对齐，而不是让它去猜。

### 相机被碰歪之后，自动对齐**可能无效**（这是设计，不是 bug）

`src/alignment.py` 用 ECC + 结构相关度对齐，**只能补偿平移/旋转/缩放**（相机被轻轻碰一下）。
一旦**取景内容本身变了**（相关度 < 0.8），它**拒绝并报告，而不是乱猜** —— 因为在规则网格上「整体挪错一个车位」和「没挪」在数学上不可分辨。

**实测过的一次**：相关度只有 0.198，强行套用一个全局位移（各行想要的位移是 +50/+5/+52/+97，**根本不是平移**）会把两行弄好、两行弄坏。**这种情况必须按 `e` 手动对齐。**

### 参考帧的语义（最容易踩的坑）

参考帧**必须是全部空置的状态**。抓帧那一刻停着的东西，之后会变成该车位**永久占用** —— 因为检测器测的是「和空置状态的差异」，而车的**消失**正好是一个差异。

顺序错了（先抓帧、后放车、再移走）就会出现「空车位报占用」。**按 `t` 做体检就能一眼看出是哪种原因。**

### 板子 3.3V 轨供不动 20 dBm 的发射电流尖峰

同固件实测：**接 LCD 185 KB/s，不接 LCD 18 KB/s，差 10 倍**（LCD 模块在充当额外去耦电容）。
**解法不需要焊电容**：把发射功率降到 10 dBm（`CONFIG_WIFI_MAX_TX_POWER_DBM`）就把差距补回来了 —— AP 就在旁边，链路预算有几十 dB 余量。

> 判断依据：**天线收发共用**，设备收到 AP 的信号是 −33 dBm（很强）说明天线好；而接收电流小、**发射才是电流尖峰**，所以只有上行方向坏（小包 0% 丢包但 RTT 66~154 ms，1.2 KB 状态页 0.19 s 正常，11.7 KB 的 TCP 帧却堵死）。

### 状态屏（P0001.2）的刷新代价

面板帧缓冲在 PSRAM，而 SPI DMA **读不了 PSRAM**，必须经内部 DMA bounce buffer。
一次性刷整帧要 ~115 KB 内部内存（板子空闲时只有 ~17~123 KB）→ **曾经直接卡死**。现在按 **16 行一带**刷（7.5 KB），并把 `trans_queue_depth` 从 10 降到 2（避免一次重绘占住 75 KB）。

### 文件读写

- **参考帧和车位配置都是原子写**（写临时文件再 `os.replace`）。因为 `cv2.imwrite` 会先截断再写，**另一个进程在这一刻读就会拿到半个 JPEG** —— 已经真实撞到过。
- **`cv2.imwrite` 失败只返回 False、不抛异常**，所以返回值必须检查。

### 不提交进 git 的东西

```
firmware/esp32-s3-cam/{build,sdkconfig,managed_components}/
tools/ota-server/{manifest.json,firmware/**/*.bin}
vision/data/{debug,reference}/, vision/data/cases/*.csv
```

**故意提交的**：`dependencies.lock`（钉住验证过的组件版本）、`vision/config/parking_spots.json`（标定是劳动成果）、`vision/data/cases/labels.json`（人工标注的真值）。

---

## 9. 目录结构

```
parking-lot-navigation/
├── project.md              ← 本文件：状态 / 工具 / 用法
├── README.md               ← 深度文档：硬件证据链、设计取舍、踩坑全过程（§15 帧率调查）
├── docs/architecture.md
├── firmware/esp32-s3-cam/  ← ESP-IDF 工程（相机节点固件）
│   ├── main/               ← 源文件（§4.1）
│   ├── partitions.csv      ← A/B 双槽布局
│   └── sdkconfig.defaults
├── vision/                 ← P0002：PC 端 OpenCV 车位占用
│   ├── src/ tests/ config/ data/
│   └── README.md
└── tools/
    ├── ota-server/         ← 薄 OTA 服务（Python 标准库）
    └── tests/              ← OTA 版本比较规则测试
```

---

## 10. 下一步

| | 内容 | 备注 |
|---|---|---|
| 1 | **看一眼 LCD 显示了什么** | P0001.2 唯一没验证的 |
| 2 | **录 5 个验收案例并拟合阈值** | `--headless --csv` → `tests/fit_threshold.py`，把 0.15 从「经验值」变成「测量值」 |
| 3 | **（可选）看情况补通道** | 现在 0 个车位离路超过 220px（引线 27~38px），**不补也行**；只有以后多了新区域、`p` 报「离路很远」时才需要 |
| 4 | **P0002.5 — 主车定位** | ESP32-S3-CAM → ArUco 标记 → `x / y / heading / visible`，填进 `parking_state.json` 的 `vehicle` 字段。**这是 P0002 封版前最后一块**：现在「停车场在哪、车位在哪、路在哪、哪个车位有车」都齐了，只差「我这辆车在哪、朝哪」 |
| 5 | **P0003 — 导航** | 拿到 `parking_map + parking_state`（含主车）后做：找可用车位 → A*/Dijkstra → Route。**不再碰视觉** |

> **P0002.5 的依赖已经就绪**：`cv2.aruco` 在当前的 `opencv-python 4.13.0` 里自带（**不需要**装 `opencv-contrib-python`，已实测）。
> 做法：打印并裁出 ArUco 标记贴在主车上 → `cv2.aruco.detectMarkers` 得到四个角点 →
> 取角点中心得 `x/y`、取一条边的角度得 `heading`、没检测到就 `visible: false` → 写进契约的 `vehicle`。
