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

**编辑模式（`e`）按键**：左键拖角/拖体、右键吸附到边、`2/4/6/8` 微调 5px、`Tab` 切换、`m` 切范围、`n` 增、`d` 删、`g`/`G` 复制整行、`a` 自动检测、`i` 内框比例、`A` 对齐、`z` 撤销、`c` 清空、`s` 保存、`q`/`ESC` 返回。
（`r` 在这一层仍是「抓参考帧」；改名用 **`R`**。）

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

## 6. 已知限制与坑

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

## 7. 目录结构

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

## 8. 下一步

| | 内容 | 备注 |
|---|---|---|
| 1 | **看一眼 LCD 显示了什么** | P0001.2 唯一没验证的 |
| 2 | **录 5 个验收案例并拟合阈值** | `--headless --csv` → `tests/fit_threshold.py`，把 0.15 从「经验值」变成「测量值」 |
| 3 | **P0003**：车位状态对外服务（signal/HTTP）→ 导航小车消费 | |
