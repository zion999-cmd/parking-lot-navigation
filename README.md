# Parking Lot Navigation / 停车场车位级导航沙盘

ESP32-S3-CAM 网络摄像头节点 —— **P0001 + P0001.1**

把一块 ESP32-S3-CAM 变成局域网里稳定的 MJPEG 网络摄像头，并带上两件基础设施：
**Wi-Fi 配网**（手机网页配置，凭据存 NVS）和**极薄 OTA**（带 A/B 分区、SHA256 校验、
原生回滚）。PC / 浏览器可以通过 HTTP 持续拿到实时画面，给后续 Python + OpenCV 消费。

**固件职责只有：拍摄 + 网络传输 + 配网 + 自升级 + 车位状态灯。** 不含 OpenCV、导航、预约、AI。

> **改过一处边界（P0008）**：前面写的「不含 LED」在加车位状态灯时被推翻了。相机节点
> 现在会**主动 GET** PC 侧的 `parking_state.json` 去点灯。这是**出站只读**，设备仍然
> 没有任何业务监听接口、不持久化业务状态——见 `docs/architecture.md` 3.6。

---

## 0. 先说怎么跑起来

PC 侧有 **4 个常驻服务**（状态服务 / 视觉 / 地址广播 / OTA），**一条命令全起来**：

```zsh
./start.sh            # 起全套；相机自己找；已经在跑的不重复起
./start.sh status     # 一屏看清谁在跑、灯带能不能工作
./start.sh stop       # 停掉
./start.sh logs vision    # 跟某个日志
```

细节见 **§21**。想不起"灯为什么不亮"的时候，先跑 `./start.sh status`。

---

## 1. 范围

```
ESP32-S3-CAM
   ├─ Camera
   ├─ Wi-Fi  (STA + SoftAP 配网)
   ├─ HTTP MJPEG
   ├─ Provisioning   ← P0001.1 新增
   ├─ OTA            ← P0001.1 新增
   └─ Bay lights     ← P0008 新增（出站读状态，不是业务接口）
```

| 做 | 不做 |
|---|---|
| Wi-Fi 配网（SoftAP + 网页 + NVS） | 配网 App / BLE 配网 |
| A/B OTA + SHA256 + 原生回滚 | 云平台 / 设备管理 / 用户系统 |
| 板载 TFT 作**状态面板**（IP / 配网 / OTA / 错误） | 屏幕上的复杂 UI、动画、地图、视频预览、触摸、菜单 |
| 车位状态灯（绿空/红占/黄导航目标，见 §17） | 接收入站业务指令（PC 推状态给设备） |
| 摄像头被碰歪后**自动对齐 ROI**（不改车位列表） | 自动增删车位 |
| 摄像头初始化（沿用已验证配置） | 视觉算法 / AI 推理 |
| HTTP MJPEG 视频流 | 私有视频协议 / 云端视频 |
| 状态页 + 串口健康日志 | 停车场业务逻辑（预约 / 计费 / 用户） |

---

## 2. 硬件

| 项目 | 值 | 来源 |
|---|---|---|
| 底板 | ESP32S3-CAM-KZB-V2.0 | 原理图 `ESP32S3-CAM-KZB-V2_0.pdf` |
| 核心板 | ESP32-S3-CAM 核心板 | 原理图 P1/P2 接口 |
| SoC | ESP32-S3 (QFN56) rev v0.2 | 运行日志 |
| CPU | 240 MHz | 运行日志 `cpu freq: 240000000 Hz` |
| PSRAM | 8 MB **八线(OCT)** @ 80 MHz，厂商 AP | 运行日志 `octal_psram: density 0x03 (64 Mbit)` |
| Flash | 16 MB，QIO，厂商 Boya | 运行日志 `spi_flash: detected chip: boya` |
| 摄像头 | **OV3660**，SCCB 地址 0x3c | 运行日志 `camera: Detected OV3660 camera` |
| USB-UART | CH340（VID 0x1a86 / PID 0x7523） | USB 描述符 |
| 串口 | UART0，GPIO43(TX)/GPIO44(RX)，115200 | 运行日志 `cpu_start: GPIO 44 and 43 are used as console UART I/O pins` |
| 车位状态灯 | **WS2812B 灯带，DIN = GPIO14**（P1 排针引出） | 见 §17 |

本项目使用 ESP32-S3 + Camera + Wi-Fi + 状态屏 + 状态灯带。底板上的麦克风、扬声器、
电池、音频均未初始化，代码里也不存在相关引用；按键只用到 BOOT（手动下载模式）。

### 2.1 ⚠️ 这块底板没有自动下载电路 —— 首次烧录必须手动

原理图里**没有** RTS/DTR 自动复位三极管电路（全文搜不到 `S8050` / NPN / RTS / DTR
网络），底板只有两个 GPIO 按键（BUTTON1=GPIO46、BUTTON2=GPIO6），
没有接到 IO0 的 BOOT 键。

实测验证：用 pySerial 穷举 `DTR`/`RTS` 的 4 种组合（谁拉 EN、IO0 低电平是 assert 还是
release），芯片**每次都被复位**（能看到 `ESP-ROM` banner）但**始终从 flash 启动**
（`boot:0x8 (SPI_FAST_FLASH_BOOT)`），没有一次进入 `boot:0x0 (DOWNLOAD)`。
结论：`RTS` 接 EN，`DTR` 未接 IO0，自动进入下载模式不可用。

所以烧录时会看到：

```
A fatal error occurred: Failed to connect to ESP32-S3:
Invalid head of packet (0x49): Possible serial noise or corruption.
```

`0x49` 就是 `'I'`、`0x44` 就是 `'D'` —— 那是设备应用日志 `I (123) ...` 的首字符，
不是 bootloader 的应答。**这不是驱动坏了，是芯片没进下载模式。**

**正确的烧录姿势**（需要用手，但很简单）：

1. 按住核心板上的 **BOOT** 键（或把 **IO0 短接到 GND**）**并保持按住**；
2. 另一个终端执行烧录命令；
3. 看到 `Connecting......` 之后变成 `Chip is ESP32-S3` 就可以松开。

```zsh
idf.py -p /dev/cu.usbserial-14220 flash monitor
```

**为什么这样就够了**：esptool 的 `default_reset` 本来就会拉 `RTS` 触发复位，
复位释放的瞬间芯片采样 IO0 —— 你手动把它拉低，芯片就进下载模式了，
不需要额外脚本。（实测确认 `RTS` → EN 有效：拉低 RTS 时芯片确实处于复位态。）

如果你更习惯先让芯片进下载模式再烧，也可以：把 IO0 接 GND → 复位 → 松开 IO0 →
立刻执行同样的命令。两种都可以。

**好消息：这只影响第一次烧录。** 装好本固件之后，设备有 OTA，
后续升级全部走网络，不用再碰线：

```zsh
# 在 PC 上
cd tools/ota-server && ./publish.sh 0.2.1
# 设备上
curl -X POST http://<device-ip>/api/ota/check     # 或在状态页点按钮
```

---

## 3. 硬件证据链（三条来源里有两条互相矛盾）

引脚映射有三份互相独立的信息源，**结论以原理图 + 真实运行日志为准**。

### 3.1 先看矛盾

| 信号 | ① 原理图 PDF | ② 参考工程 `config.h` | ③ `esp32s3-cam-kzb-v2_0-pins.md` |
|---|---|---|---|
| SIOD | GPIO4 | GPIO4 | **写成 OV_D7** |
| SIOC | GPIO5 | GPIO5 | GPIO5 ✅ |
| VSYNC | GPIO6 | GPIO6 | **完全没有** |
| HREF | GPIO7 | GPIO7 | GPIO7 ✅ |
| XCLK | GPIO15 | GPIO15 | **完全没有** |
| OV_D0 | GPIO11 | GPIO11 | GPIO11 ✅ |
| OV_D1 | GPIO9 | GPIO9 | **写成 OV_D2** |
| OV_D2 | GPIO8 | GPIO8 | **写成 OV_D3** |
| OV_D3 | GPIO10 | GPIO10 | **写成 OV_D1** |
| OV_D4 | GPIO12 | GPIO12 | GPIO12 ✅ |
| OV_D5 | GPIO18 | GPIO18 | **完全没有** |
| OV_D6 | GPIO17 | GPIO17 | **写成 TFT_DC** |
| OV_D7 | GPIO16 | GPIO16 | **写成 OV_D6** |
| PCLK | GPIO13 | GPIO13 | GPIO13 ✅ |

- **① 与 ② 完全一致：14/14 信号全对。**
- **③ 与 ①、② 严重冲突**，而且它把屏幕接口也写错了
  （它说 `CS=GPIO15, DC=GPIO20, SCL=GPIO18, BLK=GPIO19, RES=GPIO21`，
  而原理图与参考工程都是 `CS=GPIO45, DC=GPIO47, SCL=GPIO19, MOSI=GPIO20, BLK=GPIO38, RST=GPIO21`）。

**结论：`esp32s3-cam-kzb-v2_0-pins.md` 不可信。** 它缺少 VSYNC / XCLK / SIOD / OV_D5
四个必要信号，说明它不是原理图转录；文件结尾写着「如需详细功能或某一外设的专用连接说明，
请补充需求」，是 AI 生成文档的特征。**本项目不采用它的任何引脚。**

### 3.2 三条证据各自怎么来的

**① 原理图** `ESP32S3-CAM-KZB-V2_0.pdf`

原理图里摄像头信号标签和 GPIO 标签画在同一根导线上 —— 在 PDF 中它们的 y 坐标完全相同、
x 相邻。用 `pdfminer` 带坐标提取后可以无歧义配对：

```
x=204 GPIO4   ←→  x=225 SIOD        x=204 GPIO11  ←→  x=225 OV_D0
x=204 GPIO5   ←→  x=225 SIOC        x=204 GPIO12  ←→  x=225 OV_D4
x=204 GPIO6   ←→  x=225 VSYNC       x=204 GPIO13  ←→  x=225 PCLK
x=204 GPIO7   ←→  x=225 HREF        x=204 GPIO8   ←→  x=225 OV_D2
x=204 GPIO15  ←→  x=225 XCLK        x=204 GPIO9   ←→  x=225 OV_D1
x=204 GPIO16  ←→  x=225 OV_D7       x=204 GPIO10  ←→  x=225 OV_D3
x=204 GPIO17  ←→  x=225 OV_D6
x=204 GPIO18  ←→  x=225 OV_D5
```

**② 参考工程** `/Users/bx/Esp32/xiaozhi-esp32-224`

- 引脚：`main/boards/bread-compact-wifi-s3cam/config.h` 的 `CAMERA_PIN_*`
- 使用说明：`main/boards/bread-compact-wifi-s3cam/README.md`

**③ 真实运行日志**（同一块板子上跑出来的，最硬的证据）

```
I (229) Board: UUID=77bb0bfa-... SKU=bread-compact-wifi-s3cam
I (35)  octal_psram: density : 0x03 (64 Mbit)
I (78)  esp_psram: Found 8MB PSRAM device
I (81)  esp_psram: Speed: 80MHz
I (429) camera: Camera PID=0x3660 VER=0x00 MIDL=0x00 MIDH=0x00
I (429) camera: Detected OV3660 camera
I (429) camera: Detected camera at address=0x3c
I (769) ov3660: Calculated VCO: 80000000 Hz, PLLCLK: 40000000 Hz, SYSCLK: 10000000 Hz
```

日志同时满足：用的是 `bread-compact-wifi-s3cam` 这套配置，而且 OV3660 被成功识别、
成功完成 PLL 配置。**这套引脚在你手上这块板子上是跑通过的。**

### 3.3 参考工程 README 里的两处错误（不要被它误导）

1. `main/boards/bread-compact-wifi-s3cam/README.md` 写「使用的摄像头是 OV2640」，
   但真实运行日志和供应商固件名（`..._3660_...bin`）都表明是 **OV3660**。该 README 过时了。
2. 同一个 README 写「摄像头占用了 ESP32S3 的 USB 19 20 两个引脚」，
   实际占 GPIO19/20 的是**屏幕**（`DISPLAY_CLK_PIN=19`、`DISPLAY_MOSI_PIN=20`）。

### 3.4 本项目的取舍

引脚、XCLK 频率、PSRAM 配置**原样复用②**，并用①交叉验证（14/14），
再用③确认在真机上跑通。**没有换 SDK，没有重写驱动。**
唯一主动改动的是像素格式，见第 6 节。

---

## 4. 工具链（换台机器也能照着重建）

| 组件 | 版本 / 位置 |
|---|---|
| ESP-IDF | **v5.5.3** @ `/Users/bx/Esp32/esp-idf` |
| Xtensa 工具链 | `xtensa-esp-elf 14.2.0 (esp-14.2.0_20251107)` |
| 工具链根目录 | `/Users/bx/.espressif/tools/` |
| Python venv | `/Users/bx/.espressif/python_env/idf5.5_py3.11_env`（3.11） |
| cmake / ninja | 4.3.1 / 1.13.2 |
| esp32-camera | `espressif/esp32-camera` **2.1.5**（与参考工程同版本） |
| Wi-Fi 配网组件 | `78/esp-wifi-connect` **3.1.2**（MIT，参考工程同一组件） |

### ⚠️ 环境初始化必须先进 conda 环境

这是这套机器上最容易踩的坑。`export.sh` 按 PATH 上的 `python3` 版本去挑 venv，
而系统 `python3` 是 Homebrew 3.14，对应的 venv **不存在**。
必须先 `conda activate esp32`（Python 3.11），`export.sh` 才能找到 `idf5.5_py3.11_env`。

### 从零复现 build / flash / monitor

```zsh
# 1. 初始化 ESP-IDF 环境（每次开新终端都要做）
conda activate esp32
. "$HOME/Esp32/esp-idf/export.sh"

# 2. 进入固件目录
cd /Users/bx/Esp32/parking-lot-navigation/firmware/esp32-s3-cam

# 3. 编译。目标芯片不用手动选 —— sdkconfig.defaults 里有 CONFIG_IDF_TARGET，
#    IDF 会自己认出来："IDF_TARGET is not set, guessed 'esp32s3' from sdkconfig ..."
idf.py build

# 4. 查串口
ls /dev/cu.* | grep -i -E "usbserial|wchusbserial"

# 5. 烧录（⚠️ 需要先让芯片进下载模式，见 2.1 节）
idf.py -p /dev/cu.usbserial-14220 flash monitor
```

> 从零复现只需要上面这一串。`sdkconfig` 故意不提交，
> `sdkconfig.defaults` + `dependencies.lock` 才是配置的来源。

**串口名说明**：板载 CH340。macOS 自带驱动枚举成 `/dev/cu.usbserial-XXXXX`，
装了 WCH 官方驱动则枚举成 `/dev/cu.wchusbserialXXXXX`。后缀数字取决于插哪个 USB 口。

---

## 5. Wi-Fi 配网（P0001.1）

### 5.1 启动状态机

```
BOOT
 │ 读 NVS 里的 Wi-Fi 凭据
 ├─ 无凭据 ──────────────────────────► PROVISIONING
 └─ 有凭据 ─► STA_CONNECTING
                 ├─ 拿到 IP ────────► CONNECTED
                 └─ 60 秒超时 ──────► PROVISIONING
CONNECTED
 └─ 掉线 ─────► STA_RECONNECTING
                 ├─ 拿到 IP ────────► CONNECTED
                 └─ 5 分钟超时 ─────► PROVISIONING
PROVISIONING
 └─ 网页保存新凭据 ─────────────────► STA_CONNECTING
```

**永远不会卡在无限重连。** 这套流程是从参考工程
`main/boards/common/wifi_board.cc` 的 `TryWifiConnect()` / `OnWifiConnectTimeout()` /
`WifiConfigModeExit` 裁剪出来的，不是重新设计。

### 5.2 SoftAP

| 项目 | 值 |
|---|---|
| SSID | `ParkingCam-XXXX`，`XXXX` = MAC 地址最后两字节的大写十六进制 |
| 配网页面 | `http://192.168.4.1` |
| 网页来源 | 复用组件的内嵌页面（含自动跳转的 captive portal DNS） |

`XXXX` 规则来自组件实现：`snprintf(ssid, "%s-%02X%02X", prefix, mac[4], mac[5])`。

> SoftAP 是**开放网络**（组件把 `authmode` 固定为 `WIFI_AUTH_OPEN`，没有提供密码设置接口）。
> 沙盘局域网可接受：热点只暴露配网页面，不暴露视频流（配网模式下 MJPEG 服务器是停止的）。
> 想加密码必须改组件源码，本任务选择不改。

### 5.3 配网流程

1. 手机连接热点 `ParkingCam-XXXX`
2. 浏览器打开 `http://192.168.4.1`（captive portal 通常会自动弹出）
3. 页面列出扫描到的 SSID（`GET /scan`，每 5 秒刷新）
4. 选择 SSID + 输入密码 → 页面 POST `/submit`
5. 设备保存到 NVS 并连接；成功后页面跳到 `/done.html`，自动 POST `/exit`
6. 设备关闭热点、切回 STA，摄像头和视频服务器随即启动

### 5.4 NVS 凭据存储

命名空间 **`wifi`**，字符串键，最多 10 个网络：

```
ssid / password          ← 首选网络
ssid1 / password1        ← 备选
...
ssid9 / password9
```

设备扫描周围 AP，与这份列表匹配后连接，所以存多个网络可以让它自己换。
`wifi_credentials.h` 已从固件里彻底移除（`main/wifi_credentials.h.example` 只是
NVS 格式的文档说明，不参与编译）。

### 5.5 重新配网

**串口命令**（三选一里最简单可靠的一种，不需要新增硬件）：

```
clearwifi
```

通过 UART0 输入即可：清空 NVS 里的全部凭据并立即进入配网模式。
输入 `help` 看命令列表。

`idf.py monitor` 打开后会出现 `parking> ` 提示符，敲 `clearwifi` 回车即可；也可以直接写串口：

```zsh
printf 'clearwifi\n' > /dev/cu.usbserial-14220
```

实现在 `main/provisioning.cpp`，用的是 ESP-IDF 的 `esp_console` 组件
（`esp_console_new_repl_uart`）—— 它负责排空 stdout、设置行尾、安装 UART 驱动并切换到
非阻塞轮询。**不要**自己调 `uart_vfs_dev_use_driver()` 再 `fgets(stdin)`：
那只做了一半初始化，会在 newlib 的 stdio 锁上触发
`assert failed: xQueueSemaphoreTake (pxQueue->uxItemSize == 0)` 并无限重启。

---

## 6. 摄像头配置

| 项目 | 值 | 相对于参考工程 |
|---|---|---|
| sensor | **OV3660** (PID 0x3660, SCCB 0x3c) | 相同（README 说 OV2640 是错的） |
| resolution | **800x600 (SVGA)** | 参考工程用 VGA |
| pixel format | **JPEG** | 参考工程用 RGB565 |
| JPEG quality | 12 | 相同 |
| frame buffer count | **2** | 参考工程用 1 |
| frame buffer 位置 | **PSRAM** | 相同 |
| grab mode | **CAMERA_GRAB_LATEST** | 参考工程用 WHEN_EMPTY |
| XCLK | 20 MHz | 相同 |
| ledc timer/channel | LEDC_TIMER_0 / LEDC_CHANNEL_0 | 相同 |
| hmirror / vflip | 0 / 1 | 相同（参考工程注明 OV3660 需要 vflip=1） |
| PWDN / RESET | 未使用（原理图上未连接） | 相同 |

**为什么像素格式和参考工程不一样**：参考工程用 `PIXFORMAT_RGB565`，然后**在软件里**
编码成 JPEG（`main/boards/common/esp32_camera.cc` 的 `image_to_jpeg_cb`），因为它要同时
喂 LVGL 预览又要按需拍一张照片，属于「单张拍照」场景。P0001 是**连续视频流**，所以让
OV3660 直接输出 JPEG，省掉软件编码和 RGB565 字节交换 —— esp32-camera 官方 webcam 示例的
标准做法，属于「用成熟方案做系统集成」，不是新算法。**引脚和驱动没有动。**

改配置只需要改 `main/camera.c` 顶部四个宏：

```c
#define CAM_FRAME_SIZE   FRAMESIZE_SVGA   /* 改成 FRAMESIZE_VGA 可提高帧率 */
#define CAM_PIXEL_FORMAT PIXFORMAT_JPEG
#define CAM_JPEG_QUALITY 12
#define CAM_FB_COUNT     3
```

### 实测数据

> **待补**：首次烧录需要手动让芯片进下载模式（见 2.1 节），本机提交时尚未完成真机烧录，
> 因此 `observed FPS` 还没有实测值。见 P0001 验收状态。

---

## 7. 访问地址

| URL | 内容 |
|---|---|
| `http://<device-ip>/` | 状态页：IP / 传感器 / 分辨率 / FPS / RSSI / 固件版本与槽位 / 网络状态 / OTA 状态，内嵌实时画面 |
| `http://<device-ip>/stream` | 连续 MJPEG 视频流 |
| `POST http://<device-ip>/api/ota/check` | 手动触发一次 OTA 检查，返回 `202`；已有检查在跑则 `409` |
| `http://192.168.4.1/` | 配网页面（仅配网模式下存在） |

浏览器直接打开 `/` 就能看到画面。后续 Python / OpenCV 读 `/stream` 即可
（`cv2.VideoCapture("http://<device-ip>/stream")` 可以直接消费 MJPEG）。

协议就是标准的 `multipart/x-mixed-replace` + `image/jpeg`，**没有自定义私有协议**。

---

## 8. 串口启动日志

正常启动后串口会出现（P0001 的契约保持不变）：

```
WiFi connected
IP address: 192.168.3.x
Camera initialized
Stream server started
```

配网模式下则是：

```
prov: Saved Wi-Fi credentials in NVS: 0
prov: state BOOT -> PROVISIONING
prov: Wi-Fi provisioning mode
prov:   1. join the access point : ParkingCam-A1C8
prov:   2. open in a browser    : http://192.168.4.1
prov:   3. pick your network and enter its password
```

每 10 秒一行运行状态，稳定性测试和 OTA 验证都靠它：

```
uptime=600s fw=0.2.0 fps=11.8 frames=7083 clients=1 rssi=-54 dBm \
net=CONNECTED ip=192.168.3.87 heap_internal=142000 heap_psram=5200000 heap_min=131000
```

---

## 9. OTA（P0001.1）

### 9.1 分区布局

```
nvs        data nvs    0x9000    0x4000
otadata    data ota    0xd000    0x2000
phy_init   data phy    0xf000    0x1000
ota_0      app  ota_0  0x20000   0x3f0000   (3.94 MB)
ota_1      app  ota_1  0x410000  0x3f0000   (3.94 MB)
```

**≈1.21 MB 固件 / 3.94 MB 槽位 = 3.2 倍增长余量。** 尺寸与参考工程的 v2 分区表一致
（唯一的差别是不需要 8 MB 的 `assets` 分区，因此 Flash 剩余约 8 MB 未分配，留作增长空间）。

### 9.2 薄 OTA 服务

```bash
cd tools/ota-server
./run.sh                    # 端口 8000
./publish.sh 0.2.1          # 发布一个版本
```

只用 Python 标准库（`http.server`），**没有 Flask / FastAPI 依赖**。只提供两个东西：

```
GET /manifest.json
GET /firmware/<version>/parking-lot-cam.bin
```

没有用户系统、登录、数据库、设备管理、MQTT、fleet management、dashboard、遥测、
云部署。详见 [tools/ota-server/README.md](tools/ota-server/README.md)。

**manifest schema**（故意很小）：

```json
{
  "project": "parking-lot-cam",
  "version": "0.2.1",
  "firmware_url": "/firmware/0.2.1/parking-lot-cam.bin",
  "sha256": "f2d547b2...5aa5c8a",
  "size": 1270081
}
```

| 字段 | 必须 | 说明 |
|---|---|---|
| `version` | ✅ | `major.minor.patch`，按整数比较 |
| `firmware_url` | ✅ | 绝对 URL，或相对 manifest 的路径 |
| `sha256` | ✅ | 小写十六进制摘要 |
| `size` | 可选 | 给了就校验收到的字节数 |
| `project` | 可选 | 仅供参考，固件忽略 |

### 9.3 设备端行为（device pull）

```
开机 → 等 30 秒 → GET manifest
   → 版本不比当前新？ → 记 "up to date"，结束
   → 有新版本 → GET firmware（边收边算 SHA256）
        → SHA256 不符 → 丢弃，继续跑旧固件
        → SHA256 相符 → esp_ota_end → esp_ota_set_boot_partition → 重启
```

只检查一次（开机后 30 秒），之后靠 `POST /api/ota/check` 或状态页按钮手动触发。

### 9.4 版本比较规则

`major.minor.patch` 三段**按整数**比较，绝不按字典序。`0.10.0` 比 `0.9.0` 新。
可选前缀 `v` 会被忽略，缺失的段按 0 处理（`0.9` == `0.9.0`）。

实现在 `main/version.c`，**故意不依赖 ESP-IDF**，所以可以在 PC 上跑单元测试：

```zsh
./tools/tests/run.sh        # 16 项检查，含 0.10.0 vs 0.9.0
```

### 9.5 完整性与安全性

- **永远校验 SHA256**：manifest 给摘要，设备边下载边算，不符就丢弃。
  所以即使走明文 HTTP，损坏或截断的固件也不可能被装进去。
- **之后还有 ESP-IDF 自己的镜像校验**：`esp_ota_end()` 会再验一遍镜像结构。
- **HTTP vs HTTPS**：沙盘局域网用明文 HTTP + SHA256；固件也接受 `https://` URL，
  并对照 ESP x509 根证书包校验（`CONFIG_MBEDTLS_CERTIFICATE_BUNDLE=y`），
  不需要自建 CA。**不要把固件发布在不可信网络上的明文 HTTP 后面。**

### 9.6 回滚

用的是 ESP-IDF **原生回滚**，没有自己发明双分区切换协议：

```zsh
CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE=y     # 已在 sdkconfig.defaults 里
```

流程：

```
A 正常运行
  ↓ 下载 B（SHA256 通过）→ 写入空闲槽 → 设置启动分区
重启进入 B（状态 = PENDING_VERIFY，只有一次机会）
  ↓ B 跑到「确认可用」点
esp_ota_mark_app_valid_cancel_rollback()  → 状态 = VALID，保留 B
  ✗ 如果 B 在确认前崩溃/复位/断电
    → 下次启动 bootloader 把 B 标记 INVALID，回滚到 A
```

**本固件的「确认可用」点**有两个，任一到达即确认：

1. 成功进入配网模式（说明本地子系统都起来了，能对外提供服务）；
2. 连上网并成功初始化摄像头、启动视频服务器（`main/main.c` 的 `network_task`）。

刻意**不**要求必须联网才算成功 —— 否则设备在配网模式下待着就永远不确认，
下次重启会被误回滚。这两处覆盖了正常的两种工作状态；真正启动不了的固件到不了任何一处。

---

## 10. 验收方法

### 10.1 P0001（摄像头节点）

```zsh
# 1. 烧录后记录串口
idf.py -p /dev/cu.usbserial-14220 monitor | tee /tmp/stability.log
# 2. 另开终端，单客户端持续拉流 60 分钟
curl -o /dev/null --max-time 3600 http://<device-ip>/stream
```

期间看 `/tmp/stability.log` 里的 `uptime=` 是否单调递增（是否重启）、`frames=` 是否持续
增长（是否停止出帧）、`heap_min=` 是否断崖式下跌（是否内存泄漏）、`rssi=` 是否异常。

### 10.2 P0001.1（配网 + OTA）

| 场景 | 做法 |
|---|---|
| 无凭据 → 配网 | 清空 NVS（`clearwifi`）→ 重启 → 应看到 `state BOOT -> PROVISIONING` |
| 凭据失效 → 回落配网 | 保存一个不可达的 SSID → 60 秒后自动进配网 |
| 配网成功 | 手机连 `ParkingCam-XXXX` → `192.168.4.1` → 选 SSID + 密码 → 设备连上并打印 IP |
| 重启免配网 | 直接重启 → 应直接 `WiFi connected` |
| 无更新 | manifest 版本 == 当前版本 → 日志 `up to date` |
| 有新版本 | manifest 版本更新 → 下载 → `SHA256 verified` → 重启到新槽位 |
| 校验失败 | `./publish.sh --sha256 000...0 9.9.9` → 日志 `SHA256 mismatch` → 旧固件继续跑 |
| 回滚 | 推一个启动即崩的固件 → 下次启动应回到旧槽位 |

### 10.3 P0008（车位状态灯）

见 §17.6：6 条断言，其中 SC-2（`UNKNOWN` 不得显示为绿）与 SC-3（断链整条熄灭）
是必须用“停掉 `src/main.py` / 拔网线”去证的。

---

## 11. 已知限制

1. **首次烧录必须手动进下载模式**（见 2.1 节）。这块底板没有自动下载电路。
   之后所有升级走 OTA，不需要线。
2. **配网 SoftAP 是开放网络**。复用组件的固定行为，没有密码设置接口。
3. **没有 HTTPS / 鉴权**（视频流和状态页）。局域网沙盘场景可接受。OTA 有 SHA256 兜底。
4. **Wi-Fi 凭据存 NVS，明文**（设备本地）。这是 ESP-IDF 的常规做法。
5. **没有 OTA 版本签名**，只有 SHA256 完整性。manifest 本身如果被篡改，理论上可以指向
   攻击者控制的镜像（摘要也会被一起改）。局域网内可接受；生产要上 HTTPS + 签名镜像。
6. **未启用 anti-rollback**（不阻止降级）—— 沙盘需要能回退到旧版本。
7. **只有一个手动 OTA 触发方式**（HTTP + 状态页按钮），没有定时轮询。
8. **车位状态灯依赖 PC 侧的 `--host 0.0.0.0`**，而且那个 HTTP handler 没有鉴权
   （见 §17.2）。另外 5V 供电时数据线只有 0.2 V 余量（见 §17.4）。
8. FPS 统计跨客户端累计；接 N 个客户端时数值约为单客户端的 N 倍，验收测试只用 1 个客户端。
9. 麦克风 / 扬声器 / 电池 / 按键未使用，底板这些外设在本固件下不工作。
10. **状态屏会占用内部 RAM**：帧缓冲在 PSRAM，而 SPI DMA 读不了 PSRAM，驱动必须用内部
    DMA 作中转，所以屏幕刷新按 16 行分带推（每带 7.5KB）。没装屏时用
    `idf.py menuconfig` → Parking Lot Camera → 关掉 `DISPLAY_ENABLE`，显示就不会初始化、
    不分配内存、不建任务。

---

## 12. 状态显示（P0001.2）

板载 TFT 只做一件事：**把设备自身状态和“下一步要干什么”显示给人看**。
不是 UI，不做动画、地图、视频预览、触摸、菜单。

保留这块屏的核心理由：**设备 IP 肉眼可见**。换 Wi-Fi 之后不用开串口找 IP，
比赛现场和老师同学都能直接用。

### 12.1 屏幕硬件（来自参考工程实跑配置，与原理图 14/14 吐合）

| 项 | 值 |
|---|---|
| controller | **ST7789**，`esp_lcd_new_panel_st7789`（esp_lcd，不走 LVGL 驱动） |
| 分辨率 | **240x240**（参考工程 `CONFIG_LCD_ST7789_240X240=y`） |
| 总线 | **SPI3_HOST** @ 40 MHz，SPI mode 0，cmd/param 8bit，16bpp，queue depth 10 |
| 引脚 | CS=**45**，DC=**47**，RST=**21**，SCLK=**19**，MOSI=**20** |
| 背光 | **LEDC PWM** on GPIO**38**，`OUTPUT_INVERT=false`（高电平点亮） |
| 面板参数 | `invert_color=true`，`mirror(false,false)`，`swap_xy=false`，offset 0,0，RGB 元素顺序 |
| 初始化序列 | `reset` → `init` → `invert_color` → `swap_xy` → `mirror` → `disp_on_off`（与参考工程一致） |

⚠️ **LEDC 资源注意**：`camera.c` 已占用 `LEDC_TIMER_0`/`LEDC_CHANNEL_0` 产生摄像头 XCLK，
所以背光故意用 `LEDC_TIMER_1`/`LEDC_CHANNEL_1`。

### 12.2 屏幕内容

| 时机 | 显示 |
|---|---|
| 上电 | `Parking Camera` / `Booting...` / `v0.2.1` |
| 配网模式 | `Wi-Fi Setup` / `Connect to:` / `ParkingCam-XXXX` / `Then open:` / `192.168.4.1` |
| 连接中 | `Connecting Wi-Fi...` / SSID（**密码永不显示**） |
| 正常运行 | `Camera Ready` / `Wi-Fi: Connected` / `IP: 192.168.3.105` / `/stream` |
| 摄像头故障 | `Camera ERROR` / `Check camera cable` |
| Wi-Fi 不可用 | `Wi-Fi unavailable` / `Starting setup mode...` |
| OTA | `Checking update...` / `Updating... 0.2.1 -> 0.2.2 [##----] 35%` / `Update complete` / `Update failed` + `Current firmware safe` |

原则：**屏幕告诉人“该怎么办”，串口记录“程序为什么失败”。**
版本号从 `esp_app_get_description()->version` 读，没有第二份硬编码。

配网页会自动弹出，但屏幕仍然显示 `192.168.4.1`，方便现场异常时手输。

### 12.3 架构：模块只报告，不画屏

```
camera.c / wifi.cpp / provisioning.cpp / ota.c / stream.c / gate.c
        │  只调用 display_show_*()
        v
display.c   ← 唯一碰 esp_lcd / SPI 的文件
        ├─ 消息队列（定长，满了丢最旧）
        └─ display 任务：渲染 → esp_lcd_panel_draw_bitmap
```

**LEDC 不是什么“显示外设”，它是芯片里通用的 PWM 方波发生器**（名字里的 LED 只是当年的
设计用途）。它跟 WS2812 灯带**没有关系** —— 灯带走的是 **RMT**。所以按“谁占哪个
定时器/通道”来分，而不是按文件归属：

| 用 LEDC 的人 | 干什么 | 频率 | 定时器 / 通道 |
|---|---|---|---|
| `camera.c`（经 esp32-camera 的 `target/xclk.c`） | 给摄像头芯片**供 XCLK** | 20 MHz | **TIMER_0 / CH0** |
| `display.c` | 屏背光 PWM 调亮度 | 5 kHz | **TIMER_1 / CH1** |
| `gate.c`（§18） | 舵机 PWM | 50 Hz | **TIMER_2 / CH2** |
| `led.c` | **不用 LEDC**，用 RMT（1 个 TX 通道） | — | — |

**定时器决定频率，通道只是把 GPIO 接到定时器上**，所以三者各占一个不同定时器就互不影响；
S3 有 4 个定时器 / 8 个通道（`soc_caps.h:245-247`），还剩 TIMER_3 和 5 个通道。

> 早先这里写的是“display.c 是唯一碰 LEDC 的文件”，**那句话一直是不准确的** ——
> `camera.c:163-164` 从一开始就在用 LEDC TIMER_0 产生 XCLK。上面这张表才是事实。

`display_show_*()` 只往队列里投一条小消息，**任何任务都能安全调用**，
包括 Wi-Fi 事件回调。渲染、SPI 传输、背光都在 display 任务里。
屏幕是 best-effort：面板坏了、分配不到内存，固件照常做摄像头节点（SC-9）。

### 12.4 字体

`main/font8x16.h`：8x16 单色点阵，1bpp，每行 1 字节（bit7 = 最左），ASCII 0x20-0x7E。

**是光栅化出来的，不是手写的** —— 用 Pillow 把 Andale Mono 在 13px 下渲染进 8x16 单元格，
所以字形一定正确、不需要真机逐字校对。重新生成的办法写在
`font8x16.h` 头部注释里。

没选 LVGL：本项目只需要“画几行字”，把 LVGL + 显示驱动 + flush 任务拉进来
代价远大于收益，也更容易影响 streaming（SC-9）。

---

## 13. 停车位视觉 PoC（P0002）

PC 端 Python/OpenCV，独立于固件。**不改 ESP32 任何代码 —— 它只是一个视频传感器。**

```
ESP32-S3-CAM http://192.168.3.118/stream
        ↓
Python / OpenCV  →  人工标定的固定车位 ROI  →  Spot 38 EMPTY / Spot 39 OCCUPIED / Spot 40 EMPTY
```

方法：固定摄像头 + 预标定 ROI + **空车位参考帧差分**（absdiff → 阈值 → 形态学 →
changed_ratio）。不用 YOLO / 深度学习 / 透视变换。

**P0002.1 标定助手**：`Auto Detect` 用纯几何方法（CLAHE → Canny → 垂直边缘能量剖面 →
**1-D 自相关找等间距周期** → 相位最小二乘对齐 → 行分组 → 四边形）自动提出车位候选，再人工拖角修正；
默认参数是**实测标定**的（profile / vertical / band 0.05 / 周期性置信度 0.25）：
真实沙盘上得到两排 y 35..133 与 196..390、**两排间距一致 46-47px**、30 个候选；
还支持「标一格 → `g` 按几何规律生成整排」。**识别的车位数不必 100% 正确** ——
自动识别只做助手，最终 ground truth 由人工确认；失败的车位`n` 补一格再 `g` 展排即可，
`z` 可整体撤销一次自动识别。occupancy 算法一行未改。

| 实测事实 | 值 |
|---|---|
| `cv2.VideoCapture(stream)` | ❌ 打不开（FFmpeg 读不了这条 multipart 流） |
| `src/camera.py` 自写 multipart 解析 | ✅ 800x600，6 秒 60 帧 ≈ **10 fps**，0 重连 0 解码失败 |
| 画面质量 | 亮度 146.9 / 对比度 34.4 / 清晰度 Laplacian 方差 **63.5**（偏软）/ 过曝 0% / 欠曝 0% |

完整使用说明、标定步骤、阈值拟合与排错见
[`vision/README.md`](vision/README.md)。测试：

```zsh
cd vision
P=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
$P tests/test_occupancy.py        # 13 项合成图测试，无需相机
```

---

## 14. 真机调试记录（两个只有上机才发现的问题）

这两个问题在代码审阅时都看不出，都是烧到板子上才暴露的 —— 也正是 P0001 的
“SC-1~SC-7 不得凭代码审查宣称 PASS”这条规则的用处。

### 13.1 串口控制台：newlib stdio 锁断言（已修，commit `3632a87`）

```
assert failed: xQueueSemaphoreTake queue.c:1713 (pxQueue->uxItemSize == 0)
_lock_acquire_recursive  newlib/src/locks.c:174
lock_acquire_generic     newlib/src/locks.c:158
xQueueTakeMutexRecursive FreeRTOS-Kernel/queue.c:835
```

原因：手写 `uart_vfs_dev_use_driver(0)` + `fgets(stdin)` 只做了半套初始化。
修复：改用 IDF 官方 `esp_console`（`esp_console_new_repl_uart`）。

### 13.2 sys_evt 栈溢出（本次修，commit 见下）

```
***ERROR*** A stack overflow in task sys_evt has been detected.
```

原因：`provisioning.cpp` 的 Wi-Fi 事件回调里直接 `ESP_LOGI`（printf 机制吃 1~2KB 栈）
和 `wifi_rssi()`（`esp_wifi_sta_get_ap_info`），而回调跑在 **sys_evt 任务**（默认栈 3584B）里。

修复：所有入口（Wi-Fi 事件、esp_timer 回调、串口命令、公共 API）只投递定长事件，
真正干活集中到栈充裕的 `prov_task`（6144B）。

---

## 15. 帧率问题的完整调查（三个版本才定位）

现象：`/stream` 只有 1~2 fps。最终：**28.3 fps / 464~511 KB/s / 单帧发送 15.5 ms / 0 次发送失败**。

### 15.1 先让设备自己说时间花在哪

在流任务里把每帧拆成两段计时（`stream.c`），每 50 帧打一行：

```
50 frames in 1.8 s (28.3 fps) | grab avg 19.9 ms (max 30.6) | send avg 15.5 ms (max 94.8) |
jpeg avg 16.4 KB (min 14.7 max 18.5) | 464 KB/s | send failures 0
```

- `grab` = 等 `esp_camera_fb_get()` 返回一帧
- `send` = 把 JPEG 写进 socket
- 这一行是整件事的转折点。**修复前它是 `grab 16 ms / send 494 ms`** —— 97% 的时间卡在 socket 写入，摄像头有 20 倍以上余量。没有它，前几轮都在错误的地方找原因。

### 15.2 三个原因（相乘，不是相加）

**① Nagle 从来没关（最主要）**

`esp_http_server` 不设置 `TCP_NODELAY`，而一帧要分几次写（chunk 头、part 前缀、JPEG、结束符）。Nagle 会把小片段压住、等前面的数据被 ACK，再叠加对端的延迟 ACK：

- 11.7 KB 的帧，`send` 平均 **494 ms**，而链路 RTT 只有 66 ms —— 每帧白等 1~2 个往返
- 帧一大就从小减速变成**彻底堵死**：`httpd_sock_err: error in send : 11`（EAGAIN），设备写了 105 KB 而客户端只收到 35 KB

修复：在流 socket 上用 `httpd_req_to_sockfd()` + `setsockopt(TCP_NODELAY)`。**494 ms → 15.5 ms（32 倍）**。
同时把 `send_wait_timeout` 从 10 s 降到 3 s，卡死的客户端更早被放弃。

**② 发送期间一直攥着摄像头帧缓冲**

流任务在整个发送期间持有帧缓冲。`fb_count=2` 时**两个客户端就把两块缓冲全占满 → 摄像头流水线永久冻结**（日志：`frames=9` 之后再也不动，同时 `clients=2`）。

修复：先把 JPEG 拷到本任务的 PSRAM 暂存区、立刻归还摄像头缓冲，再做网络发送；`fb_count` 提到 3。

**③ 3.3V 轨供不动 20 dBm 的发射电流尖峰**

同一个固件（0.2.8）下，**接 LCD 185 KB/s，不接 LCD 18 KB/s，相差 10 倍**。LCD 模块在充当额外的去耦电容。

确定是「发射侧」而不是「天线」的判据：**天线收发共用**。设备收到 AP 的信号是 -33 dBm（很强），说明天线和接收都好；而接收电流小、**发射才是电流尖峰**，所以只有上行方向坏 —— 小包 0% 丢包但 RTT 66~154 ms（重传），1.2 KB 的状态页 0.19 s 正常，11.7 KB 的 TCP 帧却堵死。

修复：`CONFIG_WIFI_MAX_TX_POWER_DBM`（默认 10）。AP 就在旁边，链路预算有几十 dB 余量，**把发射功率降下来、尖峰电流减半就够了 —— 不需要在开发板上焊电容**。

### 15.3 OTA 一直是「GET -> HTTP 0」的原因

另一个独立的真机 bug：`esp_http_client_get_status_code()` **必须**在
`esp_http_client_fetch_headers()` **之后**调用 —— 响应头还没读进来时它返回 0。
原代码在 `esp_http_client_open()` 之后立刻取状态码，于是一律读到 0、判定失败、
不读响应体就关连接。服务器侧的表现正好是证据：

```
192.168.3.118  "GET /manifest.json HTTP/1.1" 200 -      ← 服务器回了 200
ConnectionResetError: [Errno 54] Connection reset by peer ← 设备不读响应体就关连接
```

两个调用点都修了（manifest 抓取、固件下载），后者本来也会踩同一个坑。

### 15.4 教训

- **先让设备报告时间构成**，不要在主机侧猜。`grab` vs `send` 一行就结束了前几轮的猜测。
- **一次只动一个变量**。中间有一轮同时换了固件和 LCD，导致「LCD 是主因」的错误结论，
  直到用同一固件跑出 18 KB/s vs 185 KB/s 才把两个因素分开。
- 诊断行里的 `fps=` 是**瞬时 EMA**，卡顿后连发两帧会飙到 218 —— 不要信它，看 `frames=` 的增量。

## 16. 目录结构

```
parking-lot-navigation/
├── README.md                          ← 本文件
├── .gitignore
├── docs/
│   └── architecture.md                ← 系统边界（只写已确定的部分）
├── firmware/
│   └── esp32-s3-cam/
│       ├── CMakeLists.txt             ← PROJECT_VER 在这里，即 OTA 比较的版本号
│       ├── sdkconfig.defaults         ← 目标/Flash/PSRAM/串口/回滚/OTA URL 全在这里
│       ├── partitions.csv             ← OTA A/B 布局
│       └── main/
│           ├── main.c                 ← 启动顺序、网络应用任务、健康日志
│           ├── display.c/.h           ← 状态面板（唯一碰 esp_lcd/SPI 的文件；LEDC 按定时器分，§12.3）
│           ├── font8x16.h             ← 生成的点阵字体（8x16, 1bpp）
│           ├── camera.c/.h            ← 相机（含引脚证据链注释）
│           ├── stream.c/.h            ← HTTP 状态页 + MJPEG + /api/ota/check
│           ├── provisioning.cpp/.h    ← 配网状态机 + 串口 clearwifi
│           ├── wifi.cpp/.h            ← 配网组件的 C 适配层
│           ├── ota.c/.h               ← 薄 OTA 客户端（manifest/SHA256/回滚确认）
│           ├── led.c/.h               ← 车位状态灯（唯一碰 RMT 的文件，见 §17）
│           ├── gate.c/.h              ← 闸机：触点开关 + 舵机（唯一碰 LEDC CH2 的文件，见 §18）
│           ├── version.c/.h           ← 版本比较（不依赖 IDF，可跑宿主机测试）
│           ├── idf_component.yml      ← 锁定 esp32-camera 2.1.5 + esp-wifi-connect 3.1.x
│           ├── Kconfig.projbuild      ← OTA_MANIFEST_URL + 全部 LED_* 选项
│           └── wifi_credentials.h.example  ← 仅 NVS 格式说明，不参与编译
├── vision/                            ← P0002：PC 端 OpenCV 车位占用 PoC
│   ├── README.md
│   ├── config/parking_spots.json      ← 人工标定的车位多边形
│   ├── src/{camera,spot_config,occupancy,calibration,main}.py
│   └── tests/{test_occupancy,fit_threshold}.py
└── tools/
    ├── ota-server/                    ← 薄 OTA 服务（Python 标准库）
    │   ├── server.py
    │   ├── publish.sh
    │   ├── run.sh
    │   └── README.md
    └── tests/
        ├── version_test.c             ← 版本比较宿主机测试
        └── run.sh
```

`sdkconfig` 与 `managed_components/` 不提交，`sdkconfig.defaults` 和
`dependencies.lock` 提交 —— 保证克隆下来就能得到同样的配置和同样的依赖版本。

---

## 17. 车位状态灯（P0008）

一个车位一颗 WS2812B（“2812”）像素。颜色就是全部接口，而且它是**三态语义**，
不是装饰：

| 颜色 | 状态 | 含义 |
|---|---|---|
| 🟢 绿 | `EMPTY` | 可以停 |
| 🔴 红 | `OCCUPIED` | 已占 |
| 🟡 黄 | 导航目标 | 你正在导航去的那个空车位 |
| ⚫ 灭 | `UNKNOWN` / 无数据 | 没看到，不等于空 |

### 17.1 三条不肯破的规则

1. **`UNKNOWN` 是灭的，绝不是绿的。** `parking_state/1` 就是为丢帧定义 `UNKNOWN` 的，
   `contract.py` 写得很清楚：没有新帧的车位**不得**报成 EMPTY。灯带在视觉端停止说话时
   变绿，比变暗坏得多——它在招你做一件没被确认过的事。
2. **拿不到新状态就整条熄灭。** 留上一帧的颜色是同一种谎，只是慢一点。
3. **黄灯只覆盖空位，绝不覆盖占用。** 黄的意思是“来这里”，所以
   `route.json` 里说 ok=true 但那个车位已经被占了 → **显示红**。这条不只是防御：它让两个
   状态自己互相纠正（目标被抢，黄自己变红）。

对应实现：`led.c` 的 `build_frame()`（两趟：先占位色，再用目标覆盖绿）与失败分支，
串口日志里会明说 `no fresh state - strip blanked`。

### 17.2 数据流向（新增的是两条出站只读通道）

```
车位状态  out/parking_state.json ──GET──▶ camera   绿=空位 红=占用 灭=UNKNOWN
导航目标  out/route.json         ──GET──▶ camera   黄=目标（仅当该车位为空）
          (两者都由 vision/src/serve.py 提供服务，每 500 ms)
```

设备**不接受**任何入站业务请求，也不持久化业务状态——它每 500 ms 出去看一眼，
所以 `docs/architecture.md` 3.1/3.2 的那条「单向」变成「视频出、状态入」，
而不是新开一个业务端口。

**只读两个字段，顺序不能反**：先 `ok` 再 `target_spot`。`ok=false` 也可能带着一个车位号
（`spot-not-free`、`hero-not-visible` 这些拒绝就是这样），跳过 `ok` 会让一个"你不能去"的
拒结灯成黄灯。

#### 导航结束要有人明说

`out/route.json` 原本只被写、从未被删。页面停止导航后它仍然指着最后一个目标，
于是黄灯永远不灭——一个只有灯带上看得见的、页面上早已不存在的状态。

所以有了 `GET /api/route?clear=1`（写一份 `ok=false, code=navigation-ended` 的文档），
页面在**取消导航**和**到达车位**时各调一次。到达也清，是因为车已经在那个车位上了：
黄灯的任务结束，接下来由占用检测接管（车停进去就变红）。

⚠️ **`src/serve.py` 默认只绑 127.0.0.1，设备根本拉不到。** 要让它能被拉到：

```zsh
cd vision && python3 src/serve.py --host 0.0.0.0     # 注意这个代价！
```

`--host 0.0.0.0` 会把 `--root`（整个 `vision/`，包括标定文件与 data/）暴露给局域网，
而且这个 handler **没有任何鉴权**。沙盘局域网可以接受，但要知道自己在做什么。

### 17.3 配置（全在 `idf.py menuconfig` → Parking Lot Bay Lights）

| 选项 | 默认 | 说明 |
|---|---|---|
| `LED_ENABLE` | `y` | 关掉后 `led.c` 完全不参与，`led_status()` 显示 `absent` |
| `LED_SELFTEST` | `y` | 上电自检：逐颗走灯 + 三色 + 亮度扫动。**确认后关掉** |
| `LED_GPIO` | `14` | DIN。GPIO14 是这块板上**唯一既空闲又在 P1 排针上**的脚 |
| `LED_COUNT` | `48` | 默认当车位总数。**以走灯数出来的真值为准** |
| `LED_ORDER` | `""` | 车位 id 按像素顺序，如 `9,37,B1`。**空 = 灯不工作**（会报错） |
| `LED_STATE_URL` | `http://192.168.3.102:8010/out/parking_state.json` | 车位状态（绿/红） |
| `LED_ROUTE_URL` | `http://192.168.3.102:8010/out/route.json` | 导航目标（黄） |
| `LED_POLL_MS` | `500` | 与网页的轮询节奏一致 |
| `LED_BRIGHTNESS` | `32` | 亮度**是功耗预算**，不是口味：48 颗满白 ≈ 2.9 A |

**为什么 `LED_ORDER` 不自动从 `parking_map.json` 推**：哪颗像素贴在哪条车道旁是
**安装事实**，地图不知道。想以地图顺序为起点（然后自己改）：

```zsh
python3 -c "import json;print(','.join(s['id'] for s in json.load(open('vision/out/parking_map.json'))['spots']))"
```

### 17.4 接线与电气（这一步的坑不在代码）

```
ESP32 GPIO14 ──[470Ω]── DIN（灯带的**输入**端，接错=全黑）
ESP32 GND ────────────── GND ─── 5V 电源 GND（必须共地）
5V ──[1000µF 电解，就近]── 灯带 5V
```

**已知余量不足**：灯带 5V 供电时 VIH = 0.7×5 = 3.5 V，ESP32 只能给 3.3 V，只有
0.2 V 余量。症状是第一颗像素闪/随机彩、或整条中途暗掉并伴随板子重启。
解法（不焊也能做）：把灯带供电降到 **4.3~4.5 V**；或上 74AHCT125。

这条不是纸上推导：`LED_SELFTEST` 的第三段就是亮度扫动，让这个失效模式自己暴露出来。
扫到 64/255 就停（48 颗约 0.7 A）；想故意复现掉电就把它调高，但那就是在演示中途把板子
搞重启的同一件事。

### 17.5 上机步骤

1. `python3 src/serve.py --host 0.0.0.0`，浏览器能打开
   `http://<pc>:8010/out/parking_state.json` 才继续；
2. **先只验灯带**：`LED_SELFTEST=y`（默认）已经开着，别的什么都不用配。
   `./publish.sh 0.2.12` + `POST /api/ota/check`，重启后应看到：

   ```
   led: bring-up test: count the pixels that light up, then watch for flicker
   led:   pixel 0 lit          ← 逐颗走到 LED_COUNT-1，数一下真实几颗
   led:   brightness ramp ...   ← 这一段闪/变色/板子重启 = 电平或供电不够
   led: bring-up test done
   led: LED_ORDER is empty: 48 pixels have no bay, strip stays dark
   ```

   看到逐颗走灯 = **线对、输入输出端对、供电够**。这一步最多把 `LED_COUNT` 改成
   数出来的真值；`LED_ORDER` 留空，所以测完就自己熄灭，不会冒充车位状态。
3. 填 `LED_ORDER`（车位 id 按像素顺序），再推一个版本，就开始显示真实车位状态。
4. 确认后把 `LED_SELFTEST` 关掉 —— 否则每次上电都会整条亮 10 秒。

### 17.6 验收（尚未跑）

| # | 断言 | 怎么证 |
|---|---|---|
| SC-1 | 空车位 = 绿，占用 = 红 | 手动遮一颗车位，对应像素一个轮询内变色 |
| SC-2 | `UNKNOWN` 不显示成绿 | 停掉 `src/main.py` 后重启设备：灯带**全灭**，不是全绿 |
| SC-3 | 断链不留残色 | 运行中拔 PC 网线：灯带在若干次失败后整条熄灭，日志有 `strip blanked` |
| SC-4 | 黄灯只给空位目标 | 点一个**已占**车位：应被服务端拒结（`spot-not-free`）且**不亮黄**；再手改 `route.json` 成 `ok=true,target=已占车位` → 仍显示红 |
| SC-5 | 不拖慢相机 | 健康行 `fps` 与加灯前一致，`heap_min` 不下降超过几 KB |
| SC-6 | OTA 可回滚 | 推一个坏版本（改错 `LED_GPIO`）→ 仍能从 A/B 回滚启动 |

---

## 18. 闸机（P0009）

一根触点开关说"车到了"，一颗舵机把杆抬起来；开关松开几秒后落杆。

### 18.1 为什么放在相机节点上，以及它为什么不上网

这是**本地自动化**：传感器进、执行器出，中间不经过网络。所以：

- `gate_start()` 在 **`app_main()` 里调用**，不在 `network_task()` 里 —— **PC 关机、Wi-Fi 断了，闸机照样工作**；
- 它**不读**车位状态、**不发布**任何东西，`docs/architecture.md` 的"设备只出站读"那条边界不变。

### 18.2 资源分配（每一项都有出处，不是"配置里有"）

| 资源 | 分配 | 证据 |
|---|---|---|
| **GPIO39** | 触点开关输入（对地闭合，内部上拉） | 排针引出（原理图 P1/P4）；`soc_caps.h:187,189` 确认 39 既非输入专用、也未被占用 |
| **GPIO40** | 舵机信号 | 同上 |
| **LEDC TIMER_2 / CH2** | 舵机 50 Hz，14 bit | `camera.c:163-164`=T0/CH0、`display.c:61-62`=T1/CH1 → TIMER_2 空闲；`soc_caps.h:245-247` |
| **GPIO14** | 灯带（**不动**） | 走 RMT，与 LEDC 无关 |

**已知代价**：39/40 同时是 ESP32-S3 的 JTAG 脚（`io_mux_reg.h:129-130`：39=MTCK、40=MTDO）→
**外接 JTAG 调试口没了**。崩溃回溯不受影响（`CONFIG_ESP_GDBSTUB_ENABLED=y` 走 console UART）。

### 18.3 接线

```
触点开关：  GPIO39 ────[开关]──── GND          （内部上拉，闭合=低电平=有车）
TCRT5000：  VCC=3.3V ── DO ── GPIO39 ── GND    ⚠️ 绝不能接 5V，GPIO 不耐 5V
舵机：      GPIO40 ──── 信号（橙）
            5V   ──── 舵机电源（红）← 与灯带同源，但必须满足下面 4 条
            GND  ──── 共地（黑）
```

换 TCRT5000 时**不用改代码**：两者都是"有东西=低电平"，只需确认 `GATE_SENSOR_ACTIVE_LOW`。

⚠️ **供电 4 条（这是唯一真正的风险，不是 GPIO 也不是 LEDC）**

| 负载 | 电流 |
|---|---|
| 触点开关闭合 | 0.073 mA（73 µA，可忽略） |
| 灯带正常显示 | ~170 mA |
| 灯带自检 ramp | ~720 mA |
| 舵机空载 / 转动 / **堵转** | 200-300 mA / 0.6-1 A / **1.5-2.5 A** |
| **最坏合计** | **~2.7 A** |

1. 共用那个 5V 要能持续给 **≥3A**（不能是板子的 USB 口或电池升压）
2. 舵机就近并 **1000~2200µF 电解**
3. **单点接地（星形）**：舵机电流不要从摄像头/屏幕的地线路径上过
4. 软件侧已经做了：`GATE_RAMP_MS` 慢速斜坡（不是一步到位，避免启动瞬间吸堵转电流）+ 行程只走 90° + 不撞限位

**实测症状（2026-10-16，舵机 5V 与灯带同源，硬件刚接上）**：按下开关 → 舵机转动 → **灯带整条变黑** →
几秒后舵机转回 → **灯带自己恢复，但颜色是乱的（"五颜六色"）**。

这两个现象都指向同一件事：**舵机大电流把共享的 5V 拉低了**。

- 灯带黑掉：WS2812 低于约 4V 就开始行为异常，降到 2V 以下直接不亮
- 颜色乱掉：电压不足时像素内部逻辑会**误锁存**，一个像素错位后面整条跟着错位
- 自己能恢复：`led.c` 每个轮询周期都会重发整帧（500ms 一次），所以**电压一回升它就自愈**

⚠️ 反过来说：**能持续看到乱色，说明舵机在那几秒里是"持续大电流"**，不是一瞬间的尖峰
（保持位置顶住负载也是持续扭矩）。所以**光加电容未必够，优先考虑给舵机一条独立的 5V**。

补充一条：**这跟固件无关** —— 灯带的重发逻辑是好的，是电没到位。

> 板子的 3.3V 是从同一个 5V 降下来的，所以 5V 塌了会连累摄像头和屏幕，
> 甚至出现 `rst:0x1 (POWERON)`。**闸机动作时盯串口有没有 `rst:` 就是供电够不够的实测方法。**

### 18.4 状态机与两条安全规则

```
                     ┌──── 显示二维码（GATE_QR_ON_TRIGGER）
                     ▼
CLOSED ──按下──▶ OPENING ──斜坡走完──▶ OPEN ──松开满 HOLD_MS──▶ CLOSING ──斜坡走完──▶ CLOSED
                                                                                      │
                                    二维码再留 GATE_QR_LINGER_MS，然后回状态页 ◀────────┘
```

**车到闸口 → 抬杆 + 显示二维码 → 手机扫码 → 选车位。** 这就是业务上的交接点：人正站在那儿、手里拿着车，
手机是下一个要掏的东西。

1. **一次闭合只抬一次杆**：只有 `CLOSED` 接受触发，而且必须"上一次之后见过松开"才重新武装
   （`s_armed` 初值是 `false` —— **上电时开关已经是闭合的，杆不会动**，复位场景安全）。
2. **开关还按着就不落杆，并且倒计时重置** —— 这就是防夹："车还在传感器上"和"可以落杆了"
   是同一个条件，所以它们是同一个变量，不会各说各话。**不需要第二个传感器。**

按 20 ms 一拍跑，全程非阻塞：舵机不会卡住摄像头，灯带也不会卡住舵机。
每次状态转换都打一行日志（带传感器电平和当前脉宽），健康行末尾也多一个 `gate=`。

### 18.5 舵机标定（必做，不能跳过）

"185°"通常是被改造过的 180° 舵机，末端不可靠；杆实际指哪儿还取决于舵盘怎么装的。
**所以两个位置是量出来的，不是假设的。**

```zsh
cd firmware/esp32-s3-cam
idf.py menuconfig          # → Barrier Gate → 打开 "Sweep the servo once at boot"
# 刷入后上电，舵机会以 50µs 步进慢慢从 800µs 扫到 2200µs，每步停 700ms，串口同步打印脉宽：
#     gate:   pulse  800 us (duty 655)
#     gate:   pulse  850 us (duty 696)
#     ...
# 记住"杆水平"和"杆抬起"对应的两个值（比如 1000 / 2000）
idf.py menuconfig          # 填进 GATE_PULSE_CLOSED_US / GATE_PULSE_OPEN_US，然后关掉标定
```

标定窗口从 **800µs 起**（不是 500µs）是故意的：**第一次扫一个没量过的舵机，不该有能力把它撞进限位。**

### 18.6 配置项

| 选项 | 默认 | 说明 |
|---|---|---|
| `GATE_ENABLE` | `y` | 关掉后 `gate.c` 完全不参与，健康行显示 `gate=absent` |
| `GATE_SENSOR_GPIO` / `GATE_SERVO_GPIO` | `39` / `40` | |
| `GATE_SENSOR_ACTIVE_LOW` | `y` | 对地开关和 TCRT5000 都是"触发=低" |
| `GATE_DEBOUNCE_MS` | `100` | 20ms 采样，机械抖动和手扫过都能吸收 |
| `GATE_HOLD_MS` | `5000` | **松开后**保持 5s；按着不算，且按着会重置 |
| `GATE_RAMP_MS` | `800` | 行程时长，压低启动电流尖峰 |
| `GATE_PULSE_CLOSED_US` / `_OPEN_US` | `1000` / `2000` | 90° 行程，两端都远离机械限位 |
| `GATE_START_PULSE_US` | `1500` | 上电第一个命令（中位，最不可能撞限位）；标定后建议改成落杆脉宽 |
| `GATE_CALIBRATE` | `n` | 标定模式，见 §18.5 |
| `GATE_QR_ON_TRIGGER` | `y` | 触发时显示二维码，落杆到位后按下面的时间撤下 |
| `GATE_QR_LINGER_MS` | `30000` | **杆到位后**二维码再留 30s（与杆的时序解耦，见下） |

**为什么二维码的时间要单独一个参数**：杆什么时候落是**机械问题**（车得先开出去，`GATE_HOLD_MS`=5s），
码给人看多久是**人的问题**（掏手机、对准、对焦，手里还拿着车，5~15s）。一个数字不可能同时满足两个 ——
按哪个调，另一个都是错的。默认从触发到撤码约 **36 秒**（5s 保持 + 两个斜坡 + 30s 停留），杆早就落好了，闸机也能接下一辆车。
第一次上机反馈就是「码走得比人掏手机快」，所以默认值从 10s 提到了 30s —— 只调这一个，不动杆的时序。

### 18.7 验收

| # | 断言 | 状态 |
|---|---|---|
| 1 | 原工程能构建 | ✅ 四组 Kconfig 组合（含标定路径、闸机关闭、灯带关闭）全部零警告构建通过 |
| 2 | 新增代码无警告 | ✅ 同上 |
| 3 | 开关断开时待机 | ⏳ **未实测** |
| 4 | 抖动不误触发 | ⏳ **未实测**（逻辑已实现：100ms 连续一致才采信） |
| 5 | 稳定闭合只抬一次 | ⏳ **未实测** |
| 6 | 持续闭合不重复抬杆 | ⏳ **未实测**（逻辑已实现：触发后 `s_armed=false`） |
| 7 | 松开后能重新触发 | ⏳ **未实测** |
| 8 | 抬杆→保持→落杆完整流程 | ⏳ **未实测** |
| 9 | 二维码可扫 | ⏳ 第三阶段 |
| 10 | 摄像头/屏幕/灯带无回归 | ⏳ **未实测**（构建层已验证） |
| 11 | 串口能追踪状态转换 | ⏳ 代码已实现，**未上机确认** |
| 12 | 上电/复位安全 | ⏳ **未实测**（代码已实现：`s_armed` 初值 false） |

**这张表里凡是 ⏳ 的都还没在真机上跑过** —— 构建通过不等于硬件验收，不能替代。

---

## 19. LCD 二维码

240×240 的屏上显示一个手机能扫的二维码。

### 19.1 内容与怎么改

内容是**编译期固定**的，当前是 Mac 的 mDNS 名字 + 短路径：

```zsh
# 改内容 → 重新生成矩阵 → 重编 → OTA
python3 tools/make-qr.py "http://<host>.local:8010/go" > firmware/esp32-s3-cam/main/qr_code.h
```

**地址唯一存放在生成出来的 `main/qr_code.h` 里**（`QR_PAYLOAD`）。所以 `make-qr.py` **要求把载荷当参数传**、
不设默认值 —— 有默认值就等于多一份地址副本，两份副本早晚有一份是旧的（这个项目已经吃过两次这个亏）。

**为什么要 `/go` 这个短地址**：屏上模块像素数由载荷长度决定，短一个版本就大一圈：

| 载荷 | 字符 | QR 版本 | 每模块 | 实际效果 |
|---|---|---|---|---|
| `<名字>.local:8010/web/navigation.html` | 55 | 4（33 模块）| **5px** | QR 165px，留白 18px |
| `<名字>.local:8010/go` ← **当前** | **38** | **3（29 模块）** | **6px** | QR 174px，留白 33px |
| ~~`192.168.3.102:8010/go`~~ | 28 | 2（25 模块）| 7px | 但**换个 Wi-Fi 就废** |

**用名字不用 IP，代价正好是一个版本（7px → 6px）**，换回来的是"**换 Wi-Fi 也不用重编**"。
这个交易是划算的：一颗每次搬场地都要重新生成的二维码，比一颗小 14% 的二维码坏得多。

**`/go` 这个短路径在这里更要紧**：直接写 `/web/navigation.html` 会掉到 **5px/模块**，
多这 17 个字符是买一个版本，不是省事（代价只是 `serve.py` 里 6 行 302，页面仍然只有一份）。

**为什么是"离线生成矩阵"而不是在固件里跑编码器**：载荷大约一个安装点只改一次，编码器就是为已经决定的事
白占 flash。代价是改地址要重编 + OTA，而这件事是"几分钟、做一次"。
矩阵 **625 位 = 79 字节**，渲染就是往已有 framebuffer 里填方块 —— **零依赖、零运行时代码、零内存分配**
（满足"避免在高频刷新路径中重复编码、反复分配内存"）。

生成脚本放在 `tools/` 里，和 `make-aruco-marker.py` 同一个理由：产出常数的那东西，应该和常数放在一起。

### 19.2 尺寸：模块像素数是唯一重要的量

240px 的屏，二维码要能扫，**模块像素数**是唯一重要的量；静区（白边）是编码规范的一部分，不能省。
缩放倍数是**算出来的**（`display.c` 的 `draw_qr()`），不是写死的：

```
版本 3 = 29 模块，静区 4 模块 → 29 + 2×4 = 37 模块
240 ÷ 37 = 6.48  → 取 6px/模块  → QR 174×174px，四边留白 33px（= 5.5 模块 ✅）
```

**再大就只能牺牲静区**，而"大 12% 但扫不出来"等于没放大。换算表：

| 版本 | 模块 | 总宽（含静区）| 每模块 | QR 尺寸 |
|---|---|---|---|---|
| 1 | 21 | 29 | 8px | 168px |
| 2 | 25 | 33 | 7px | 175px |
| **3** | **29** | **37** | **6px** | **174px** |

有意思的是**几个版本的物理尺寸几乎一样**（168 / 175 / 174px）—— 决定成败的是模块大小 8 → 7 → 6。
所以"把网址缩短"不是省字符，是**把每个模块放大约 17%**。

纠错等级选 **L** 也是为这个：换 **M** 会把版本 3 推到版本 4，模块从 6px 掉到 5px。
在一块干净的 LCD 上，**模块大小比纠错冗余更值钱**。

> 屏上**不加文字标注**：240px 里放不下"能扫的码 + 能读的字"，而码是有活要干的那个。
> 载荷在串口日志里打（`display: showing QR code: "PARKING-TEST-001" (21 x 21 modules)`），
> 排查"扫不出来"时要对照的正是它。

### 19.3 什么时候显示

| 主触发（P0009） | **车到闸口按下触点开关 → 显示二维码**；杆落到位后再留 `GATE_QR_LINGER_MS`（默认 30s）撤下。从触发算起约 **36 秒** |
|---|---|---|
| 开机自检用 | 上电也显示 `QR_BOOT_SECONDS`（默认 **45 秒**），专门用来验证"这码到底扫不扫得出来" |

两者是独立的开关：`GATE_QR_ON_TRIGGER` 和 `QR_SHOW_ON_BOOT`。**确认能扫之后，把 `QR_SHOW_ON_BOOT` 关掉**，
以后就只由闸口触发。

- QR 是**临时覆盖层**（和 OTA 那几个屏同类）：`display_restore()` 会把底下的状态页放回来，
  不用谁去记住"原来显示的是啥"
- **在 Wi-Fi 连上之前就显示**：一个只在网络健康时才出现的二维码，正好没法用来排查网络问题
- 45 秒不是随便写的：一个人**手里还拿着小车**的情况下，"掏手机 → 开相机 → 对准 → 对焦"实测要 5~15 秒，
  45 秒留了三次机会

### 19.4 验收

| # | 断言 | 状态 |
|---|---|---|
| 1 | LCD 正常显示二维码 | ✅ **实测**（2026-10-16 上机确认屏上出现） |
| 2 | 手机能扫出完整内容 | ✅ **实测**（扫出 `parking-test-001` —— 当时载荷就是那个测试串） |
| 3 | 扫出来的地址能打开页面 | ⏳ **换成 URL 后待复测**（第一次扫出的是明文串不是网址，见下） |
| 4 | 原有 LCD 背光/摄像头/灯带/闸机不受影响 | ⏳ **未实测**（构建层已验：7 组配置零警告） |

**第一次扫码得到的证据**：扫出来是 `parking-test-001`（**全小写**），而载荷里写的是大写
`PARKING-TEST-001`。这说明**扫码器把它当主机名处理并做了小写归一化**（iOS 的典型行为）——
也就是说 **码本身是能读的、读得准的**，只是内容不是网址。载荷换成 `http://...` 之后这条路就通了。

⚠️ **手机必须和 PC 在同一个局域网**：载荷是内网地址（`192.168.3.x`），不在同一网里扫了也打不开。

⚠️ **PC 的 IP 是 DHCP 的，而它被烧进了二维码**：IP 一变，码就是废的。要么在路由器上给 PC 做**静态绑定**，
要么把载荷换成主机名（`http://bxs-macbook-pro-7.local:8010/go`，39 字符 → 版本 3 → 6px 模块），
用 mDNS 换掉对固定 IP 的依赖 —— 代价是每模块小 14%。

**已离线验证的部分**（不依赖硬件）：
- 生成的 63 字节矩阵与 `qrcode` 库的矩阵**逐位比对 441/441 全一致**
- 几何计算：8px/模块 + 4.5 模块静区（> 规范要求的 4 模块）
- ASCII 预览确认三个定位图案、时序图案、完整静区都在

---

## 20. 服务器地址动态发现（P0011）

### 20.1 问题：地址写死在固件里，换个 Wi-Fi 就全废

原本有 **6 处**写死的地址，而且是**两个方向**：

| 方向 | 位置 | 内容 |
|---|---|---|
| 设备 → PC（4 处）| `Kconfig` 的 OTA / 状态 / 目标 URL + `qr_code.h` 的二维码 | PC 的 IP |
| PC → 设备（2 处）| `vision/src/main.py:554`、`calibration.py:1007` 的 `--source` | 设备的 IP |

把沙盘搬到另一个 Wi-Fi，PC 的 IP 一变：**灯带拉不到状态会全灭、OTA 拉不到清单、二维码指向不存在的地址**，
而唯一的修法是重编固件。这跟"搬到现场就能用"是矛盾的。

### 20.2 做法：PC 广播，设备听

```
tools/beacon.py（PC，每 2s）
    UDP 广播 → 255.255.255.255:9999
    "PARKINGLOT/1 {...}"                 ← 内容只给人看
                    ↓
设备（discovery.c）：只听，认源地址 ← 这个地址"能到达设备"是被证明过的
                    ↓
把三个 URL 的 host 换掉（端口和路径不动）
```

```zsh
# PC 侧（和 serve.py 一起常驻）
python3 tools/beacon.py                    # 默认 2s 一次
python3 tools/beacon.py --to 192.168.3.255 # 某些网络需要定向广播
```

**为什么用 UDP 广播而不是 mDNS**：不用引组件（mDNS 是实打实的 flash 成本）、不用注册服务、不用配路由器，
**搬到任何 Wi-Fi 都不用改任何东西**。沙盘本来就在一个子网里，广播够用。
（代价：广播不过路由器；访客网络开了客户端隔离的话，本来 HTTP 也过不去，不是这个方案特有的限制。）

**为什么只用源地址、不读包里的 host 字段**：包里的 host 会是"同一个事实的第二份真相"。
**到达设备的那个地址**才是被证实的那个。

### 20.3 哪些跟随广播，哪些不跟随（这是刻意的）

| 用途 | 跟随广播？ | 理由 |
|---|---|---|
| 车位状态 / 导航目标 | ✅ 跟随 | 最坏情况只是**灯的颜色不对** |
| **OTA** | ❌ **默认不跟随**（`DISCOVERY_APPLY_OTA`，默认 `n`）| 最坏情况是**执行陌生人的固件**。而唯一的校验是同一份 manifest 提供的 SHA256（README §11.5：**没有签名**）。局域网里为"灯的颜色"冒这个险可以，为"代码执行"不行 |
| 二维码 | ❌ 仍烧死在固件里 | 见下 |

### 20.4 二维码：改指 Mac 的 mDNS 名字（已完成）

二维码里原本烧着 PC 的 IP，换 Wi-Fi 就废。现在改成 **Mac 的 mDNS 名字**：

```
http://bxs-macbook-pro-7.local:8010/go     ← 里面没有任何 IP
```

**为什么这个方案最省**：macOS 的 Bonjour **本来就**在广播自己的 `.local` 名字，所以
**固件端一行 mDNS 代码都不用加**（不需要 `espressif/mdns`，不需要设备端再做转发）。
手机直接问 Mac，设备完全不在这条路径上。

> 我最初描述 (b) 时说"编**设备**的名字、设备 302 转发" —— 那需要固件里加 mDNS 应答器（约 60–80KB flash）。
> **实际验证（手机能打开 `bxs-macbook-pro-7.local:8010`）指向的是更省的那一条**：编 Mac 的名字。
> 区别就在这里：谁的名字，决定固件要不要动。

**代价**：名字比 IP 长 → 二维码掉一个版本（7px/模块 → 6px/模块）。若要换回 7px，
可以把 Mac 的 LocalHostName 改短（`sudo scutil --set LocalHostName parking-pc`，30 字符 → 版本 2），
代价是改了这台 Mac 在 AirDrop / 文件共享里显示的名字。

**两个前提**：
- 手机必须能解析 `.local`（你的手机验证过 ✅；换别的手机演示前先试一次）
- **名字必须稳定**：局域网上若出现同名 Mac，macOS 会自动把自己改成 `xxx-2.local` → 二维码就废了

### 20.4.1 还剩一处没动态（PC 侧的设备地址）

设备 → PC 的 4 处已经全部解决（3 个 URL 跟随广播 + 二维码不含地址）。
剩下 **PC → 设备** 的 2 处：`main.py:554` 与 `calibration.py:1007` 的 `--source` 默认值里写着设备 IP。

**这个其实可以白拿**：设备每次来拉状态文档时，PC 的 HTTP 服务**天然看到它的源 IP**。
所以 `serve.py` 记下最后一个来拉的客户端地址，`main.py` 用 `--source auto` 去读 ——
**设备的轮询本身就是它的信标**，PC 侧再也不用维护设备地址。（还没做）

### 20.5 端口和路径不跟着变

`urlhost.c` 只改 host，**端口、路径一律不动**。理由：服务换个端口就是另一个服务，那是**配置**；
而 host 是**环境**。只有环境会随场地变。

```
http://192.168.3.102:8010/out/parking_state.json  +  10.0.0.7
→ http://10.0.0.7:8010/out/parking_state.json       ← 换环境靠这一行
```

### 20.6 失效时的行为：退回固件里配的地址

- **没听到广播**（beacon 没跑 / 刚开机）→ 用 `Kconfig` 里配的地址，和以前完全一样
- **广播超过 15 秒没来** → 当成"那台机器不在了"，退回配置值，而不是一整天对着一个已经离开的机器说话
- 所以 `DISCOVERY_ENABLE` **可以一直开着**，它只是改善，不是依赖

### 20.7 验证

| 项 | 结果 |
|---|---|
| URL 改写的边界（无 scheme、host 非法、放不下、https、带查询/片段/无端口…）| ✅ **22 项宿主机测试全过**（`tools/tests/run.sh`，和 `version.c` 同一套）|
| PC 广播真的能发出去并被收到 | ✅ 当场验证：4 秒收到 5 个包，源地址 `192.168.3.102`、魔数正确 |
| 六组 Kconfig 组合零警告构建 | ✅ 且大小各异（关掉发现 −1008 B）|
| **设备真的能听到并切换服务器** | ✅ **实测**（2026-10-16）：`discovery: no beacon heard; using the configured addresses` → Wi-Fi 连上后 → `discovery: server is 192.168.3.102 (beacon: PARKINGLOT/1 {"host":...})`。**退回配置值和跟随广播两条路径都验到了** |

---

## 21. 一条命令起全套（`start.sh`）

PC 侧要常驻 4 个进程，靠人记是记不住的，所以有一个脚本：

```zsh
./start.sh                 # 起全套（默认命令就是 start）
./start.sh status          # 一屏看清状态
./start.sh stop            # 停掉
./start.sh restart
./start.sh logs vision     # 跟日志；名字有 serve / vision / beacon / ota
```

### 21.1 四个服务，各自是干什么的

| 服务 | 端口 | 少了它会怎样 |
|---|---|---|
| **`serve.py`** | 8010 | **灯带全灭**（拿不到状态文档）、**二维码扫了跳不过去**、网页打不开 |
| **`main.py`**（视觉）| 8020 | `parking_state.json` 不再更新 → **灯带按设计全灭** |
| **`beacon.py`** | 9999 | 设备退回固件里配的地址 → **换 Wi-Fi 后要重编固件** |
| `ota-server` | 8000 | 平时没影响；**升级时**拉不到 manifest |

`main.py` 用的是带 OpenCV 的那个 python（conda 的 `vision-dev` 环境）。
挪了环境就用 `PY_VISION=/path/to/python ./start.sh` 覆盖。

### 21.2 `status` 会告诉你什么

```
stack status
  serve    up (pid 85941)
  vision   up (pid 86764)
  beacon   up (pid 85927)
  ota      up (pid 85916)

  what the strip needs
  state document: 8 s old, 48 bays          ← 超过 15 秒就会报警（灯带会灭）
  bays match the calibration                ← 不匹配就报警，见下
  10 occupied, 38 empty, 0 unknown
  camera: http://192.168.3.118/stream  (Firmware 0.2.20 (slot ota_1) )
  beacon: running - the device will follow this PC's address
```

**`occupied` / `empty` / `unknown` 是分开数的**，因为它们在灯带上不是一回事：
**EMPTY 是绿，UNKNOWN 是灭**。把它们合成一个数字，就看不出"停车场是空的"和
"检测器还没看到这些车位"的区别了（刚启动时全是 UNKNOWN 是正常的，灯带会先灭一下）。

**"状态文档和标定不一致"那一条是特意加的**：`main.py` 是用启动时的标定跑着的，
改完地图不重启它，状态文档里就是**另一套车位 id** —— 灯带会一颗都匹配不上、**整条黑掉，
而且日志里一行都没有**（因为"灭→灭"没有变化可打）。这个故障真发生过，所以现在脚本直接报出来。

### 21.3 相机地址是脚本自己找的

`main.py` 的 `--source` 默认值里写着一个设备 IP，那个地址会变。所以脚本：

1. 先看缓存 `run/camera.txt`（快，一次请求验证）
2. 缓存无效就**扫本机 /24**，按**状态页标题** `Parking Lot Camera` 认设备（不是按 IP 认）
3. 找到就记进缓存，下次直接用

所以平时**不用管设备叫什么地址**。要指定就 `./start.sh --source http://<ip>/stream`。

> 这一步把"PC → 设备"那处写死的地址在**使用层面**解掉了；`main.py` 里的默认值还在，
> 但已经不需要人去改它。（彻底去掉的做法见 §20.4.1。）

### 21.4 不会重复启动

已经在跑的服务（**包括别人手动起的**）会被识别出来并**"收养"**（记下 pid），不会再起一个：
两个 `serve.py` 抢 8010、两个 `main.py` 抢摄像头，都是很难查的故障。

日志和 pid 都在 `run/`（已 gitignore）：`run/serve.log`、`run/vision.log` …

### 21.5 现象 → 查哪个服务

| 现象 | 先查 |
|---|---|
| 灯带全灭 | `serve.py`（:8010）→ 再看状态文档是否新鲜、id 是否与标定一致 |
| 二维码扫出来打不开 | `serve.py`（:8010）在不在；手机是否和 PC 同网；Mac 的 `.local` 名字能不能解析 |
| 升级没反应 / `manifest: fetch failed` | `ota-server`（:8000） |
| 换了 Wi-Fi 后什么都要重编 | `beacon.py` 没跑 |
| 找不到相机 | 板子上电没、在同一 Wi-Fi 没；或 `--source` 指定 |
| 完全不知道从哪查 | `./start.sh status` |

### 21.6 写这个脚本时踩出来的四个坑（都修了，记下来）

| 坑 | 症状 | 处理 |
|---|---|---|
| `set -e` + `pipefail` 把"没找到"当致命错误 | `status` 跑到一半就退出（`pgrep` 没匹配、`curl` 连不上**都是正常答案**）| 这类调用全部显式标记"这不是失败" |
| vision 的"就绪"没有判据 | 刚起完就报 `! stale: main.py is not writing` ——**把正常启动中的栈诊断成坏的** | 就绪判据改成"**状态文档真的变新鲜**"（`state_fresh`），等不到就把它的日志打出来 |
| `EMPTY` 和 `UNKNOWN` 混成一个数字 | 显示 `0 occupied, 48 empty/unknown`，看不出"停车场是空的"和"检测器还没看到"的区别（**EMPTY 是绿、UNKNOWN 是灭**）| 分开数，并且对"刚启动时的 UNKNOWN"单独解释 |
| 相机健康检查超时太紧 | 报 `camera does not answer`，**而 vision 正在读它的流** | 相机在推流时链路被塞满（ping 从 2ms 涨到 70-95ms），单次状态请求可能超过 2 秒 → 超时放宽到 4 秒，缓存失败先**重试一次**（8 秒）再重新探测 |

> 最后一条的教训值得单独说：**在状态检查里报假警报，比不报还坏** —— 它会把排查引到一个正在正常工作的设备上。
> 所以"不答"这种结论，宁可多等几秒、多试一次。

### 21.7 ⚠️ OTA 死循环：一次真实事故（2026-10-16）

**现象**：设备不停"启动 → 检查更新 → 下载 → 重启 → 再启动"，进度条走完照样重复。
OTA 服务日志里能看到**同一个固件被反复下载**（那次是 12 次）。

**根因**：`discovery_start()` 被放在了 `wifi_init()` **之前**。`wifi_init()` 才会跑
`esp_netif_init()` → `tcpip_init()`；在那之前调 lwIP 的 `socket()` 是在跟一个还不存在的协议栈说话：

```
assert failed: tcpip_send_msg_wait_sem …/lwip/src/api/tcpip.c:454 (Invalid mbox)
Backtrace: netconn_apimsg -> tcpip_send_msg_wait_sem -> __assert_func
```

`addr2line` 解析 backtrace 确认了调用链。

**为什么会变成死循环**（这是设计层面的坑，不是那一个 bug）：

```
旧固件(0.2.20) 查到新版本 → 下载 → 写另一个槽 → 重启进新槽(0.2.22)
   新固件在"确认自己有效"之前 panic（因为它连 Wi-Fi 都没起来）
   → 引导程序按设计回滚到 0.2.20
   → 0.2.20 又查到"有更新的版本" → 又下载 → 无限循环
```

**A/B 回滚机制本身是对的**（它确实挡住了一个坏镜像，没让设备变砖）。
问题是"**回滚 + 开机自动检查 + 不记得这个版本已经失败过**"三者相加 = 无限循环。

**修法**：把 `discovery_start()` 移到 `wifi_init()` 之后（`main.c`），并在两处都写明这个顺序约束和它导致的事故。

**实测验证**（见上一节 §20.7）：0.2.23 装上去、自己确认有效、循环终止。

**还没做的护栏**：让设备记住"这个版本试过、没活下来"，下次不再试。
现在一旦推上去的固件在启动早期有问题，就会**反复刷写同一个分区**（那次 12 次），
且只能靠停 OTA 服务或插 USB 才能停下。这条护栏能把"无限循环"变成"有界失败"。

### 21.8 说明

- 脚本按 **macOS 自带的 bash 3.2** 写（不用关联数组、不用 `${var,,}`、不用 `mapfile`）
- 端口想改就改脚本顶部的常量；`serve.py` 绑的是 `0.0.0.0`（**必须**，设备和手机是别的机器）
- 这个脚本是**开发/演示用的**，不是部署工具：没有守护、没有开机自启、崩了不会重拉
