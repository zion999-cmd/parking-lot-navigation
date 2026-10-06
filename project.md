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
| **P0002** | PC 端车位占用检测 PoC（OpenCV） | ✅ **已可用** | 48 个车位、空/占用识别正确、静止场景下 48/48 稳定为 0 差异 |

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
| 测试 | **86 项全绿**（occupancy 17 / detector 17 / editor 37 / alignment 15） |

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
| `tests/demo_alignment.py` | 对齐演示 / 测当前偏移 | `$P tests/demo_alignment.py --shift 0,0 --frame X.jpg` |
| `tests/fit_threshold.py` | **从验收案例拟合阈值**（阈值要测量，不要猜） | `$P tests/fit_threshold.py --cases 'data/cases/*.csv'` |

### 4.4 测试

```zsh
tools/tests/run.sh                       # OTA 版本比较规则（宿主测试，纯 C）
cd vision && for t in tests/test_*.py; do $P $t; done    # 86 项
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
| **⑤ 标道路**（导航层，可选） | 编辑模式里按 **`t`** 或点顶部 **[ Road Network ]** → 沿路连续点击（自动连成中心线）→ `k` 改节点类型 → `b` 绑车位 → `s` 保存 |

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

**编辑模式（`e`）顶部有两个页签：`[ Parking Spots ] [ Road Network ]`**（按 `t` 切换，或直接点）。

**Parking Spots 层**（摄像头的问题）：左键拖角/拖体、右键吸附到边、`2/4/6/8` 微调 5px、`Tab` 切换、`m` 切移动范围、`n` 增、`d` 删、`g`/`G` 复制整行、`a` 自动检测、`i` 内框比例、`A` 对齐、`z` 撤销、`c` 清空、`s` 保存。
（`r` 在这一层仍是「抓参考帧」；改名用 **`R`**。）

**Road Network 层**（导航的问题，**纯手工，不做自动检测**）：

| 操作 | 键/鼠标 |
|---|---|
| 放节点并连成中心线 | **连续点击**（每个新节点自动连到上一个） |
| 移动节点 | 拖动，或 `2/4/6/8` |
| 分支 / 汇入 | **右键点一个已有节点**（从当前链头接过去） |
| 另起一条路 | `n`（断开链条），或右键点空白 |
| 节点类型 | `k` 循环：入口 → 路口 → 路径点 → 车位接入点 |
| 绑定车位（单个） | `b`（作用于选中的节点；重复按会改绑，会提示从哪个改到哪个，`z` 可撤销） |
| **连接所有车位** | **`B`（Connect Bays）—— 见下面的推荐流程** |
| 删除节点 | `d`（连同它的边一起删） |
| 检查 | `p`（报告悬挂边、不存在的车位号、还没连接的车位） |
| 其他 | `z` 撤销　`c` 清空　`s` 保存　`Tab` 下一个节点 |

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

## 6. 停车场地图：两层结构

```
停车场地图（一个文件，两个图层）
│
├── Parking Spots     ← 摄像头的问题：这个车位空着吗？
│     ├── Spot 38  polygon
│     └── ... 48 个
│
└── Road Network      ← 导航的问题：车该怎么开过去？
      ├── nodes  入口 / 路口 / 路径点 / 车位接入点
      └── edges  节点之间的连接（可单向）
```

**为什么道路不做成一个大 polygon**：道路的**视觉边界对导航不重要**，真正重要的是**车应该沿着哪些位置走** —— 那是中心线，是人的决定，不是能检测出来的东西。所以道路层**完全不参与占用识别，也不做自动检测**。

**为什么两图合一个文件**：地图的两个图层如果分开存，就会出现「车位说 800x600、道路说别的尺寸」或者只更新了一边的情况。

```json
{
  "comment": "...",
  "image_size": [600, 800],
  "spots": [ { "id": "38", "polygon": [[...]] } ],              ← 车位层
  "nodes": [ { "id": "n1", "kind": "entrance", "x": 60, "y": 300 } ],  ← 道路层
  "edges": [ { "a": "n1", "b": "n2" } ]
}
```

**两层互不破坏是代码保证的属性，不是口头约定**：每个图层保存时，会**把它在文件里读到的所有段落原样重新写出**（`map_document.build()`）。写道路不会丢车位，写车位不会丢道路 —— 两层各有测试守着（`tests/test_road.py`）。

**节点类型**（`kind`）：

| kind | 含义 | 画面上 |
|---|---|---|
| `entrance` | 入口 | 绿点 `IN` |
| `junction` | 路口 | 黄点 `X` |
| `waypoint` | 路径点（塑造路径的弯） | 灰点 `.` |
| `spot` | **车位接入点**（携带车位号） | 橙点 `P38` |

两个图层**始终同时绘制**（当前层实色、另一层淡色）—— 因为在道路层里，车位框是「接入点该落在哪里」的参照。

### 怎么标（推荐流程）

**不需要手点 48 个车位接入点。** 最小可用路网 = 1 个入口 + 每条通道上几个点：

1. `e` 进编辑模式 → `t` 切到 **Road Network**
2. **沿通道连续点击**画出中心线（入口 → 路口 → 路口）—— **一行车位画一条通道**
3. 按 **`B`（Connect Bays）**
4. 按 **`p`** 检查：一开始会报「47 个车位还没道路连接」，归零就说明导航能到每一个车位

**`B` 做了什么**：给**每个还没有接入点的车位**，在「**朝向最近那段路的那条边的中点**」放一个 `spot` 节点，连到路上；如果接入点落在某条边的中间，还会**自动把那条边拆成两段**插一个路口进去（否则一个 16 车位的长通道上，接入点会被拽到角落，支线斜穿其他车位）。

**这是推导，不是检测**：「车位朝哪边开口」由两个**人的决定**推出——车位多边形 + 你画的路。`B` 只是把它们接起来，**完全不看画面**（在道路层按 `a` 它会明确拒绝自动识别）。

**幂等**：以后新增了车位，再按一次 `B` 只会连新的那些。

**通道要画在车位附近**：接入点到通道的距离就是车要开的最后一段。实测用一条横跨整个画面的通道，支线中位数 135px、最长 298px（穿过多排车位）；**一行一条通道**支线就只剩几十像素。

### 导航接口（给 P0003）

```python
net = RoadNetwork.load("config/parking_spots.json")
net.adjacency()   # {节点: [(邻居, 像素距离), ...]}，单向边只出现一次 → A*/Dijkstra 直接可用
net.spot_nodes()  # {"38": "n8"} → 车位号到接入点节点号
net.validate([...])  # 悬挂边 / 不存在的车位号 / 还没连接的车位
```

实测：48 个车位 + 一条通道 → **99 节点 / 98 边，从入口 Dijkstra 可达 48/48 个车位**。

**还没做的**：单向边（`oneway`）模型、校验、邻接表都支持，但 UI 里还没有切换它的键 —— 需要时直接手改 JSON 里的 `"oneway": true`。

**文件位置**：就是 `vision/config/parking_spots.json`（名字是历史遗留，它现在装的是整张地图）。`main.py --config` 默认指向它，视觉代码只读 `spots` 段、忽略不认识的段落，所以两层共用一个文件不需要改视觉代码。

---

## 7. 已知限制与坑

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

## 8. 目录结构

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

## 9. 下一步

| | 内容 | 备注 |
|---|---|---|
| 1 | **看一眼 LCD 显示了什么** | P0001.2 唯一没验证的 |
| 2 | **录 5 个验收案例并拟合阈值** | `--headless --csv` → `tests/fit_threshold.py`，把 0.15 从「经验值」变成「测量值」 |
| 3 | **把道路路网标出来** | `e` → `t` → 沿路点击 → `b` 绑车位；48 个车位都连上之后 `p` 就不再报告未连接 |
| 4 | **P0003**：车位状态对外服务（signal/HTTP）→ 导航小车消费 | 导航侧直接用 `adjacency()` + `spot_nodes()` |
