# UHF RFID（4D 四天线模块）

一个 4 端口的 UHF（EPC Gen2 / ISO 18000-6C）读写模块，配 USB-TTL 接 PC，后续接 ESP32。
用途：**拿到标签数据**（EPC / RSSI / 天线号），不是精确定位。

## 0. 现状：实测过什么、没测过什么

| 项 | 状态 |
|---|---|
| 串口通信、协议、CRC | ✅ 实测：版本 `R204_4 V1.3` / `V3.13`，每帧 CRC 正确 |
| 读标签 | ✅ 实测：三张标签，10 次轮询读到 8 次 |
| 通知帧**带天线号** | ✅ 实测：`PL=18`，最后一字节为天线号 |
| **4 个天线端口单独工作** | ✅ 实测：逐端口选通都能读 |
| 天线敏感区 | ✅ 实测：**约 30 cm**（与供货商一致） |
| **天线间串扰** | ❌ **未测**（决定能不能按天线分车位） |
| 功率对串扰的影响 | ❌ 未测 |
| 接 ESP32 | ❌ 未做 |

## 1. 硬件与接线

### 模块参数（厂商数据手册）

```
协议        EPCglobal UHF Class 1 Gen 2 / ISO 18000-6C
频段        840–960 MHz
供电        DC +3～5V，高功率下需稳定 +5V
电流        待机 <10mA，发射峰值 ≈300mA
输出功率    5–28 dBm，1 dB 步进，默认 26 dBm
串口        TTL UART，**3.3V**，115200 8N1
射频接口    SMA / IPEX ×4
尺寸        23×20×3 mm
```

⚠️ **`EN` 管脚必须 > 1.1V 模块才工作**（15pin FPC 第 14 脚 / 5pin 第 5 脚）。

⚠️ **先接天线再发射** —— 空载发射是这类模块最常见的损坏原因。

### 接 PC（当前阶段）

```
USB-TTL TX  →  模块 UART_RXD
USB-TTL RX  →  模块 UART_TXD
USB-TTL GND →  模块 GND
模块 VCC    →  5V（≥500 mA，与适配器共地）
```

⚠️ **"TX 接 TX" 是完全静默** —— 排查时先试反接。

### 将来接 ESP32

模块 UART 是 **3.3V TTL** ✓ 可以直接接；需要一路**空闲 UART**（43/44 是控制台、41/42 是地磁的 I2C，都不要占）。

## 2. 协议（厂商《通讯协议使用说明》V4.1.3）

```
BB | Type | Command | PL(MSB) PL(LSB) | Parameters | Checksum | 7E
```

- `Type`：`0x00` 命令 / `0x01` 响应 / `0x02` 通知
- `PL`：**两个字节**，高位在前
- `Checksum`：从 Type 到最后一个参数累加，取低字节

### 三条基本命令（已逐字节对照手册验证）

```
读硬件版本   BB 00 03 00 01 00 04 7E
读软件版本   BB 00 03 00 01 01 05 7E
单次轮询     BB 00 22 00 00 22 7E        ← 轮询前后自动开关功放
选择天线 n   BB 00 1B 00 02 01 <n> <crc> 7E      （n = 1..4）
多天线盘点   BB 00 1B 00 05 02 <a1><a2><a3><a4> <crc> 7E
设置功率     BB 00 B6 00 08 <p1><p2><p3><p4> <crc> 7E   （功率 = dBm × 100）
```

### 标签通知帧（关键）

```
BB 02 22 00 12 <RSSI> <PC hi lo> <EPC 12> <CRC 2> <ANT> <sum> 7E
             PL = 0x12 = 18
RSSI   有符号字节，dBm（如 0xBF = -65）
PC     0x3400 = 96 bit EPC
ANT    **天线号，0 基**（0..3 对应端口 1..4）
```

⚠️ 单天线版本 PL = 17（无 ANT 字节）。**必须按 PL 长度判断**，否则会把 CRC 当 RSSI 读。

### 无标签

```
BB 01 FF 00 01 15 16 7E      错误码 0x15 = 场内无标签 / 标签 CRC 错
```

## 3. 工具

| 工具 | 用途 |
|---|---|
| `tools/rfid/e310_probe.py --port X` | 一次性诊断：读版本、验证链路、`--loopback` 回环自测（短接 TX/RX）|
| `tools/rfid/e310_live.py --port X` | **实时**：每 200 ms 轮询，按 `q` 退出，退出时给累计表；`--antenna n` 选端口，`--cycle` 轮流 4 路，`--csv f.csv` 记录 |
| `tools/rfid/e310_antenna_matrix.py --port X` | **串扰矩阵**：逐个天线读，列出每个 EPC 被哪些天线听到、强度差多少 dB |

```bash
python tools/rfid/e310_probe.py --port /dev/cu.usbserial-14330
python tools/rfid/e310_live.py  --port /dev/cu.usbserial-14330 --antenna 1
python tools/rfid/e310_antenna_matrix.py --port /dev/cu.usbserial-14330 --rounds 4
```

## 4. 能做什么、不能做什么

**能** ✓（已实测）

- 读到场内**有哪些标签**（EPC，防冲突支持多标签）
- 每个标签的 **RSSI**（粗略反映远近）
- 每次读到的**天线号**（4 个端口之一）

**不能** ✗（硬件限制，已和需求方确认）

- ❌ **不做三边定位** —— 这块模块**没有相位/RSSI 测距能力**，RSSI 只能给"远/近"的粗分
- ❌ **不能按天线精确分车位** —— 敏感区约 **30 cm**，远大于沙盘单个车位，一根天线会读到附近多张标签（**串扰待测**）

**因此沙盘里的定位职责划分**：

```
车位级"有没有车"      → 相机（现有，已冻结在 hero-v1-static 基线）
"场上有哪些标签"      → RFID（本文档）
"标签在哪个区域"      → 天线号，前提是串扰足够小（待测）
```

## 5. 待测（按重要性）

1. **串扰矩阵**：4 张标签分别放 4 根天线前，看是否只被自己那根读到、强度差多少 dB。
   判据：**spread ≥ 10 dB** 才算能按天线分区。
2. **功率扫描**：26 / 20 / 15 / 10 dBm 各测一次串扰。**降功率通常是减小串扰最有效的手段**，30 cm 的敏感区在低功率下会明显缩小。
3. **读取距离曲线**：标签从贴天线移到读不到，记 RSSI 与失读点。
4. **多标签稳定性**：4 张同时在场时的读取率（防冲突）。

## 6. 踩过的坑

| 坑 | 现象 | 原因 |
|---|---|---|
| "TX 接 TX" | 完全静默，8 个波特率都没反应 | 丝印 `TXD` 是**模块自己的发送**，要接适配器 RX |
| 以为只有 1 个天线可用 | 只有端口 1 能读 | 天线要**先选通**（`BB 00 1B 00 02 01 <n>`），默认只开端口 1 |
| 把 PL 当 1 字节 | 帧解析全部错位 | **PL 是 2 字节** |
| 按 PL=17 解析 | 有标签但判定"没读到" | 四天线固件 PL=18，多一个 ANT 字节 |
| USB 口过流 | 适配器从系统里消失 | 换线时误碰 VCC 短路，拔插复位即可 |

## 7. 参考

厂商资料（本地）：`~/Downloads/4D资料/`

- `超高频模块多天线4D数据手册.pdf` —— 电气参数、管脚定义
- `超高频读写模块多天线系列通讯配置协议/RFID模块通讯协议使用说明.pdf` —— 完整协议（57 页）
- `四端口快捷指令.txt` —— 天线选择 / 功率设置（GBK 编码）

## 8. 接到 ESP32-S3 的接线方案（已定稿）

### 最终接线

```
RFID 模块                ESP32-S3
──────────────────────────────────────────
GND      ───────────►    GND
VCC      ───────────►    5V (VCC5V)      ⚠️ 峰值 ≈300 mA
TX       ───────────►    GPIO1   (UART1 RX)
RX       ◄───────────    GPIO2   (UART1 TX)
EN       ───────────►    3.3V            （模块要求 >1.1V）
```

UART1 @ 115200 8N1。**GPIO1 = 模块 TX 进 ESP32**，**GPIO2 = ESP32 发给模块**。

### 引脚占用（从源码与 sdkconfig 核出来的，不是猜的）

| 资源 | GPIO | 来源 |
|---|---|---|
| Camera | 4 (SIOD), 5 (SIOC), 6 (VSYNC), 7 (HREF), 8–12 (D2,D1,D3,D0,D4), 13 (PCLK), 15 (XCLK), 16–18 (D7,D6,D5) | `main/camera.c:42-55` |
| LCD | 19 (SCLK), 20 (MOSI), 21 (RST), 38 (BL), 45 (CS), 47 (DC) | `main/display.c:51-56` |
| WS2812 灯带 | 14 | `CONFIG_LED_GPIO=14`，`main/led.c:157` |
| 红外开关 / 道闸地感 | 39 | `CONFIG_GATE_SENSOR_GPIO=39`，`main/gate.c:379` |
| 舵机 | 40（LEDC TIMER_2 / CH2）| `CONFIG_GATE_SERVO_GPIO=40`，`main/gate.c:424` |
| QMC5883 I2C | 41 (SDA), 42 (SCL) | `CONFIG_MAG_SDA_GPIO/SCL_GPIO`，`main/mag.c:605` |
| UART0 控制台 / 烧录 / 日志 | 43 (TX), 44 (RX) | `CONFIG_ESP_CONSOLE_UART_NUM=0`（非 CUSTOM，即默认引脚）|
| SPI flash | 26–32 | 芯片固定 |
| Octal PSRAM | 33–37 | `CONFIG_SPIRAM_MODE_OCT=y` |

**空闲**：`0, 1, 2, 3, 46, 48`（ESP32-S3 没有 GPIO22–25）

### 为什么选 1 / 2

| 判据 | GPIO1 / GPIO2 |
|---|---|
| 工程内引用 | 无（`grep GPIO_NUM_\|_PIN\|_GPIO` 全仓库无匹配）|
| strapping 脚 | 不是（strapping 是 0 / 3 / 45 / 46）|
| 双向能力 | 是（见下）|
| 影响烧录 / 日志 | 不影响，UART0 未动 |
| 与现有外设冲突 | 无 |

`GPIO48` 留作备用；`GPIO46` 留到真的没有普通脚时才考虑。

### 关于 GPIO46 与 "输入专用" —— 一处更正

**ESP32-S3 没有输入专用的 GPIO。** 芯片能力定义里两者完全相同：

```c
// components/soc/esp32s3/include/soc/soc_caps.h
#define SOC_GPIO_VALID_GPIO_MASK        (0x1FFFFFFFFFFFFULL & ~(BIT22|BIT23|BIT24|BIT25))
#define SOC_GPIO_VALID_OUTPUT_GPIO_MASK (SOC_GPIO_VALID_GPIO_MASK)   // 相等
```

所以 GPIO46 **可以输出**，可以做 UART TX。早先文档里"46 是输入专用"的说法是错的 —— 那是 **ESP32-S2** 的 GPIO46，套到 S3 上不成立。

**但 GPIO46 确实是 strapping 脚**（S3 的 strapping 是 **GPIO0 / GPIO3 / GPIO45 / GPIO46**）。它参与启动模式判定：默认 SPI Boot 下 GPIO0=1 时 GPIO46 可为任意值；若 GPIO0 被拉低进入下载相关模式，GPIO46 的电平就参与判定。因此它**不是不能用，而是上电时不能乱给电平** —— 接一个空闲高电平的 UART TX 通常无碍，但没有必要冒这个险，因为 1/2/48 是真空闲。

### 电平与供电

- 模块 UART 是 **TTL 3.3V**（数据手册原文："UART 串口接收管脚，TTL 3.3V"）→ **与 ESP32 直连，不需要电平转换**
- **VCC 必须 5V**（"高功率下需稳定 +5V"），峰值 ≈300 mA
- ⚠️ README §17 记录过：**舵机动作会把共享 5V 拉低**，导致灯带整条变黑。RFID 发射同样吃这条轨 —— 若出现"舵机一动 RFID 掉线"，给 RFID 独立 5V 或就近加 1000 µF

### 固件计划（未开始）

1. `main/rfid.c`：UART1（RX=GPIO1 / TX=GPIO2，Kconfig 可配），115200 8N1，**best-effort**（失败只记日志，不影响相机/灯带/道闸）
2. 帧解析：`BB … 7E`，**PL 为 2 字节**，校验和验证
3. 轮询任务：默认 200 ms 一次单次轮询，维护 `EPC → {last_seen, rssi, antenna}`
4. `GET /api/rfid`，与 `/api/mag` 同一风格
5. Kconfig 菜单 "UHF RFID"：`RFID_ENABLE / RFID_UART / RFID_RX_GPIO / RFID_TX_GPIO / RFID_POLL_MS / RFID_ANTENNA`

⚠️ **不要占 43/44（控制台）和 41/42（地磁 I2C）**。本工程**无法使用外部 JTAG**（MTCK/MDTI/MTDO/MTMS = 39/41/40/42 全被 IR、地磁、舵机占用），**也无法使用原生 USB**（D-/D+ = 19/20 被 LCD 的 SCLK/MOSI 占用）→ **43/44 是唯一的调试通道**，更不能挪用。
