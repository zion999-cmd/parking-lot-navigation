# Parking Lot Navigation / 停车场车位级导航沙盘

P0001 — **ESP32-S3-CAM 网络摄像头节点**

把一块 ESP32-S3-CAM 变成局域网里稳定的 MJPEG 网络摄像头。PC / 浏览器可以通过 HTTP
持续拿到实时画面，后续给 Python + OpenCV 消费。

**本阶段只有一件事：拍摄 + 网络传输。** 不包含 OpenCV、导航、预约、LED 等业务。

---

## 1. P0001 范围

```
Camera → ESP32-S3 → Wi-Fi → HTTP → PC / Browser
```

| 做 | 不做 |
|---|---|
| Wi-Fi STA 连接 + DHCP | 配网 App / SoftAP 配网 / BLE 配网 |
| 摄像头初始化（沿用已验证配置） | 视觉算法 / AI 推理 |
| HTTP MJPEG 视频流 | 私有视频协议 |
| 状态页 + 串口健康日志 | 停车场业务逻辑 |

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

本项目**只使用 ESP32-S3 + Camera + Wi-Fi**。底板上的 TFT、麦克风、扬声器、电池、
按键、音频均未初始化，代码里也不存在相关引用。

---

## 3. 硬件证据链（重要 —— 三条来源里有两条互相矛盾）

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
x=204 GPIO4   ←→  x=225 SIOD
x=204 GPIO5   ←→  x=225 SIOC
x=204 GPIO6   ←→  x=225 VSYNC
x=204 GPIO7   ←→  x=225 HREF
x=204 GPIO15  ←→  x=225 XCLK
x=204 GPIO16  ←→  x=225 OV_D7
x=204 GPIO17  ←→  x=225 OV_D6
x=204 GPIO18  ←→  x=225 OV_D5
x=204 GPIO8   ←→  x=225 OV_D2
x=204 GPIO9   ←→  x=225 OV_D1
x=204 GPIO10  ←→  x=225 OV_D3
x=204 GPIO11  ←→  x=225 OV_D0
x=204 GPIO12  ←→  x=225 OV_D4
x=204 GPIO13  ←→  x=225 PCLK
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
唯一主动改动的是像素格式，见第 8 节。

---

## 4. 工具链（换台机器也能照着重建）

| 组件 | 版本 / 位置 |
|---|---|
| ESP-IDF | **v5.5.3** @ `/Users/bx/Esp32/esp-idf` |
| Xtensa 工具链 | `xtensa-esp-elf 14.2.0 (esp-14.2.0_20251107)` |
| 工具链根目录 | `/Users/bx/.espressif/tools/` |
| Python venv | `/Users/bx/.espressif/python_env/idf5.5_py3.11_env`（3.11） |
| cmake / ninja | 4.3.1 / 1.13.2 |
| esp32-camera | `espressif/esp32-camera` **2.1.5**（与参考工程同版本，见 `dependencies.lock`） |

### ⚠️ 环境初始化必须先进 conda 环境

这是这套机器上最容易踩的坑。`export.sh` 按 PATH 上的 `python3` 版本去挑 venv，
而系统 `python3` 是 Homebrew 3.14，对应的 venv **不存在**。
必须先 `conda activate esp32`（Python 3.11），`export.sh` 才能找到 `idf5.5_py3.11_env`。

或者用一个函数包起来（加进 `~/.zshrc`，可选）：

```zsh
get_idf() {
  conda activate esp32
  . "$HOME/Esp32/esp-idf/export.sh"
}
```

### 从零复现 build / flash / monitor

```zsh
# 1. 初始化 ESP-IDF 环境（每次开新终端都要做）
conda activate esp32
. "$HOME/Esp32/esp-idf/export.sh"

# 2. 进入固件目录
cd /Users/bx/Esp32/parking-lot-navigation/firmware/esp32-s3-cam

# 3. 编译
#    目标芯片不用手动选 —— sdkconfig.defaults 里有 CONFIG_IDF_TARGET，
#    IDF 会自己认出来：
#    "IDF_TARGET is not set, guessed 'esp32s3' from sdkconfig .../sdkconfig.defaults"
#    只在你确实想改目标芯片时才需要 idf.py set-target esp32s3
idf.py build

# 5. 查串口（CH340 会同时出现 usbserial-* 和 wchusbserial* 两种命名）
ls /dev/cu.* | grep -i -E "usbserial|wchusbserial"

# 6. 烧录 + 监视
idf.py -p /dev/cu.usbserial-14220 flash monitor

# 只烧录：
idf.py -p /dev/cu.usbserial-14220 flash
# 只看串口：
idf.py -p /dev/cu.usbserial-14220 monitor
```

> 从零复现只需要上面这一串。实测：删掉 `sdkconfig` 和 `build/` 后直接
> `idf.py build` 就能成功（已验证，产物大小 `0xdd1a0` 与首次构建一致）。
> `sdkconfig` 故意不提交，`sdkconfig.defaults` + `dependencies.lock` 才是配置的来源。

退出 monitor 用 `Ctrl+]`。

**串口名说明**：板载 CH340。macOS 自带驱动会枚举成 `/dev/cu.usbserial-XXXXX`，
装了 WCH 官方驱动则枚举成 `/dev/cu.wchusbserialXXXXX`。两者都能用。
后缀数字取决于**插在哪个 USB 口**，换口就会变。
如果报 `port is busy or doesn't exist`，先 `ls /dev/cu.*` 重新确认端口号。

---

## 5. Wi-Fi 配置

STA 模式，连接现有路由器，DHCP 取地址。

凭据放在**被 gitignore 的本地头文件**里，不会进 Git：

```zsh
cd firmware/esp32-s3-cam/main
cp wifi_credentials.h.example wifi_credentials.h
# 编辑 wifi_credentials.h，填上 WIFI_SSID / WIFI_PASSWORD
```

`wifi_credentials.h` 在 `.gitignore` 第 1 行。
如果忘了创建这个文件，编译会直接报错并提示你怎么做（不会静默用错密码）。

本项目**不实现**配网 App、SoftAP 配网、BLE 配网。

---

## 6. 访问地址

设备拿到 DHCP 地址后（串口会打印 `IP address: x.x.x.x`）：

| URL | 内容 |
|---|---|
| `http://<device-ip>/` | 状态页（含实时画面） |
| `http://<device-ip>/stream` | 连续 MJPEG 视频流 |

浏览器直接打开 `/` 就能看到画面。后续 Python / OpenCV 读 `/stream` 即可
（`cv2.VideoCapture("http://<device-ip>/stream")` 可以直接消费 MJPEG）。

协议就是标准的 `multipart/x-mixed-replace` + `image/jpeg`，**没有自定义私有协议**。

---

## 7. 串口启动日志（验收要求）

正常启动后串口会依次出现：

```
WiFi connected
IP address: 192.168.1.x
Camera initialized
Stream server started
```

另外每 10 秒打印一行运行状态，稳定性测试就靠它：

```
uptime=600s fps=11.8 frames=7083 clients=1 rssi=-54 dBm heap_internal=142000 heap_psram=5200000 heap_min=131000
```

字段含义：上电秒数（判断是否重启）、帧率、累计帧数（判断是否停止出帧）、
客户端数、信号强度（判断是否断 Wi-Fi）、内部内存 / PSRAM 剩余 / 历史最小剩余
（判断是否内存泄漏）。

---

## 8. 摄像头配置

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
| xclk ledc timer/channel | LEDC_TIMER_0 / LEDC_CHANNEL_0 | 相同 |
| hmirror / vflip | 0 / 1 | 相同（参考工程注明 OV3660 需要 vflip=1） |
| PWDN / RESET | 未使用（原理图上未连接） | 相同 |

### 为什么像素格式和参考工程不一样

参考工程用 `PIXFORMAT_RGB565`，然后**在软件里**编码成 JPEG
（`main/boards/common/esp32_camera.cc` 里的 `image_to_jpeg_cb`）。
它这么做是因为要同时喂 LVGL 预览、又要按需拍一张照片，属于「单张拍照」场景。

P0001 是**连续视频流**，所以让 OV3660 直接输出 JPEG，省掉软件编码和 RGB565 字节交换。
这是 esp32-camera 官方 webcam 示例的标准做法，属于「用成熟方案做系统集成」，
不是新算法。**引脚和驱动没有动。**

分辨率优先级按需求取 `稳定性 > 延迟 > 帧率 > 分辨率`，从 SVGA 起步（在建议的 VGA/SVGA 区间内），
没有一上来就 UXGA / 高 quality / 极限 FPS。

改配置只需要改 `main/camera.c` 里这四个宏，都在文件顶部：

```c
#define CAM_FRAME_SIZE   FRAMESIZE_SVGA   /* 改成 FRAMESIZE_VGA 可提高帧率 */
#define CAM_PIXEL_FORMAT PIXFORMAT_JPEG
#define CAM_JPEG_QUALITY 12
#define CAM_FB_COUNT     2
```

### 实测数据

> **待补**：本机提交时板子不在 USB 上，尚未做真机运行，因此
> `observed FPS` / 稳定性时长还没有实测值。见第 10 节。

---

## 9. 稳定性验收方法

最低 30 分钟，目标 60 分钟：

```zsh
# 1. 烧录后开始记录串口
idf.py -p /dev/cu.usbserial-14220 monitor | tee /tmp/stability.log

# 2. 另开一个终端，用单客户端持续拉流 60 分钟
#    （-o /dev/null 丢弃数据，只测量能否持续收到）
curl -o /dev/null --max-time 3600 http://<device-ip>/stream
```

期间检查 `/tmp/stability.log` 里的 `uptime=` 是否单调递增（是否重启）、
`frames=` 是否持续增长（是否停止出帧）、`heap_min=` 是否断崖式下跌（是否内存泄漏）、
`rssi=` 是否异常（是否断 Wi-Fi）。

`stream latency` 用主观判断即可 —— 这是沙盘摄像头，不做视频编码性能研究。

---

## 10. 已知限制

1. **未做 OTA**。分区表只有一个 `factory` 槽（4 MB），升级必须重新烧录。
   本项目不需要 OTA，做了反而增加复杂度。
2. **Wi-Fi 凭据在编译期**写进固件。换 Wi-Fi 要重新编译烧录。
   这是需求允许的最简机制，代价是灵活性。
3. **没有 HTTPS / 鉴权**。任何能访问该 IP 的人都能看到画面。局域网沙盘场景可接受。
4. **FPS 统计是跨客户端累计的**。接 N 个客户端时数值约为单客户端的 N 倍，
   验收测试只用 1 个客户端。
5. **Wi-Fi 失败不重启**，会继续初始化摄像头并启动服务器，等关联上之后即可访问。
   摄像头或 HTTP 服务器失败则会 5 秒后重启（这两个是本地必需件）。
6. 参考工程那份 `esp32s3-cam-kzb-v2_0-pins.md` 引脚文档是错的，见第 3 节。
7. 屏幕 / 麦克风 / 扬声器 / 电池 / 按键未使用，因此底板这些外设在本固件下不工作。

---

## 11. 目录结构

```
parking-lot-navigation/
├── README.md                          ← 本文件
├── .gitignore                         ← 屏蔽 wifi_credentials.h
├── docs/
│   └── architecture.md                ← 系统边界（只写已确定的部分）
└── firmware/
    └── esp32-s3-cam/
        ├── CMakeLists.txt
        ├── sdkconfig.defaults         ← 目标/Flash/PSRAM/串口 全部在这里
        ├── partitions.csv             ← nvs + phy_init + factory(4MB)
        └── main/
            ├── CMakeLists.txt
            ├── idf_component.yml      ← 锁定 esp32-camera 2.1.5
            ├── main.c                 ← 启动顺序 + 健康日志
            ├── camera.c / camera.h    ← 相机（含引脚证据链注释）
            ├── wifi.c / wifi.h        ← STA 连接 + 自动重连
            ├── stream.c / stream.h    ← HTTP 状态页 + MJPEG
            └── wifi_credentials.h.example   ← 模板（真正那份被 gitignore）
```

`sdkconfig` 与 `managed_components/` 不提交，`sdkconfig.defaults` 和
`dependencies.lock` 提交 —— 保证克隆下来就能得到同样的配置和同样的依赖版本。
