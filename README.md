# Parking Lot Navigation / 停车场车位级导航沙盘

ESP32-S3-CAM 网络摄像头节点 —— **P0001 + P0001.1**

把一块 ESP32-S3-CAM 变成局域网里稳定的 MJPEG 网络摄像头，并带上两件基础设施：
**Wi-Fi 配网**（手机网页配置，凭据存 NVS）和**极薄 OTA**（带 A/B 分区、SHA256 校验、
原生回滚）。PC / 浏览器可以通过 HTTP 持续拿到实时画面，给后续 Python + OpenCV 消费。

**固件职责只有：拍摄 + 网络传输 + 配网 + 自升级。** 不含 OpenCV、导航、预约、LED、AI。

---

## 1. 范围

```
ESP32-S3-CAM
   ├─ Camera
   ├─ Wi-Fi  (STA + SoftAP 配网)
   ├─ HTTP MJPEG
   ├─ Provisioning   ← P0001.1 新增
   └─ OTA            ← P0001.1 新增
```

| 做 | 不做 |
|---|---|
| Wi-Fi 配网（SoftAP + 网页 + NVS） | 配网 App / BLE 配网 |
| A/B OTA + SHA256 + 原生回滚 | 云平台 / 设备管理 / 用户系统 |
| 板载 TFT 作**状态面板**（IP / 配网 / OTA / 错误） | 屏幕上的复杂 UI、动画、地图、视频预览、触摸、菜单 |
| 摄像头初始化（沿用已验证配置） | 视觉算法 / AI 推理 |
| HTTP MJPEG 视频流 | 私有视频协议 / 云端视频 |
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
#define CAM_FB_COUNT     2
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
8. FPS 统计跨客户端累计；接 N 个客户端时数值约为单客户端的 N 倍，验收测试只用 1 个客户端。
9. 屏幕 / 麦克风 / 扬声器 / 电池 / 按键未使用，底板这些外设在本固件下不工作。

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
camera.c / wifi.cpp / provisioning.cpp / ota.c / stream.c
        │  只调用 display_show_*()
        v
display.c   ← 唯一碰 esp_lcd / SPI / LEDC 的文件
        ├─ 消息队列（定长，满了丢最旧）
        └─ display 任务：渲染 → esp_lcd_panel_draw_bitmap
```

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

## 15. 目录结构

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
│           ├── display.c/.h           ← 状态面板（唯一碰 esp_lcd/SPI/LEDC 的文件）
│           ├── font8x16.h             ← 生成的点阵字体（8x16, 1bpp）
│           ├── camera.c/.h            ← 相机（含引脚证据链注释）
│           ├── stream.c/.h            ← HTTP 状态页 + MJPEG + /api/ota/check
│           ├── provisioning.cpp/.h    ← 配网状态机 + 串口 clearwifi
│           ├── wifi.cpp/.h            ← 配网组件的 C 适配层
│           ├── ota.c/.h               ← 薄 OTA 客户端（manifest/SHA256/回滚确认）
│           ├── version.c/.h           ← 版本比较（不依赖 IDF，可跑宿主机测试）
│           ├── idf_component.yml      ← 锁定 esp32-camera 2.1.5 + esp-wifi-connect 3.1.x
│           ├── Kconfig.projbuild      ← OTA_MANIFEST_URL
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
