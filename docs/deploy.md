# 从零部署：这个仓库里有什么、还需要自己做什么

仓库里有全部**源码**，但**不包含任何一台具体设备的现场数据**。直接在别处 `git clone` 之后，
代码能编译、单测能跑，但系统**不会立刻工作** —— 缺三样必须现场生成的东西。

本文说清这三样是什么、怎么生成、以及漏了会看到什么现象。

---

## 1. 仓库里有什么 / 没有什么

| 项 | 在仓库 | 说明 |
|---|---|---|
| 固件源码（相机/LCD/灯带/道闸/地磁/RFID） | ✅ | `firmware/esp32-s3-cam/main/` |
| PC 视觉代码 + 356 个单测 | ✅ | `vision/src/`、`vision/tests/` |
| HOG 模型（历史资产）| ✅ | `vision/data/hero_hog.yml`、`tools/hero_lab/hero_hog.yml`。**运行时不再加载** —— HERO 的位置来自 HSV+shape，朝向来自 `red_arrow`；HOG 已退出执行路径，保留只为可复现那段否决记录（见 `docs/hero-pipeline.md`）|
| HSV 标记参考轮廓 | ✅ | `vision/data/hsv_marker/reference_arrow.json` |
| 车位多边形（默认标定） | ✅ | `vision/config/parking_spots.json` —— 是**作者那张桌子**的标定 |
| **占用检测参考图** | ❌ | `vision/data/reference/reference.jpg` **必须自己拍**（唯一真正的运行时缺失）|
| **地磁标定样本** | ❌ | `vision/data/hsv_marker/calibration.jsonl`，按需重新采 |
| **HOG 训练数据集** | ❌ | `tools/hero_lab/data/`，重新训练才需要 |

被排除的原因：参考图、标定、数据集都是**某一台设备的测量结果**，不是源码，跟着仓库走只会误导。
（见 `.gitignore` 里的注释。）

---

## 2. 新克隆后必须做的三件事

### 2.1 拍参考图（不做则占用检测完全失效）

```bash
./start.sh edit          # 打开实时窗口，车库画面
# 把 48 个车位全部清空，然后按 r
```

生成 `vision/data/reference/reference.jpg`。

⚠️ **漏了会看到**：48 个车位全部 `changed_ratio = 0.000`、`state = EMPTY`，无论车上不上桌。
程序里 `ref MISSING - press r` 就是这个意思。

⚠️ **拍错会看到**：如果拍的时候桌上有车，之后所有车位都读 0（参考图里已经有车了）。
恢复办法：清空车位，重新按 `r`。

### 2.2 确认或重标车位多边形

```bash
./start.sh edit          # 切到 [ Parking Spots ] 层
```

- `a` Auto Detect 只**增加**候选，不会动已有工作
- `A` 把全部车位对齐到当前帧（相机漂移补偿，不增不删）
- 拖角、整体移动、`s` 保存

标定结果写进 `vision/config/parking_spots.json`（**这个文件仓库里有默认值，但几乎一定需要按你的桌子重标**）。

### 2.3 烧固件

```bash
cd firmware/esp32-s3-cam
idf.py build
idf.py -p /dev/cu.usbserial-XXXX flash      # 串口烧录
# 或者走 OTA：
cd ../.. && ./tools/ota-server/publish.sh 0.2.73
```

⚠️ **两件事**：
- 板子的 USB 口有两个（标 `TTL` 的是 UART0 桥，可用于烧录和日志；标 `OTG` 的是原生 USB，但
  **GPIO19/20 被 LCD 占用**，所以 OTG 口只能供电）
- `sdkconfig` 是**本地生成**的，不入库；`sdkconfig.defaults` 入库。第一次 build 会按 defaults 生成。

---

## 3. 启动与验收

```bash
./start.sh start --source http://<板子IP>/stream
./start.sh status
```

应看到：

```
serve    up        # 页面 http://127.0.0.1:8010
vision   up        # 主循环
ota      up        # 固件发布
state document: 0 s old, 48 bays
bays match the calibration
N occupied, M empty, 0 unknown
```

验收清单：

- [ ] `vision/out/parking_state.json` 每秒刷新
- [ ] `curl -s http://127.0.0.1:8010/out/parking_state.json` 里 `vehicle` 有位置
- [ ] `GET /api/mag`（板子）返回 4 条地磁
- [ ] `GET /api/rfid`（板子）返回标签表 + 每口计数
- [ ] `python -m pytest vision/tests -q` → 356 passed

---

## 4. 其余子命令

```bash
./start.sh edit                        # 前台跑 main.py（标定/看细节）
./start.sh label                       # 标注工具 http://127.0.0.1:8030
./start.sh logs vision                 # 看日志
./start.sh stop                        # 全停
```

`label` 有三种标定模式（下拉框）：

- **车框（4 角）** —— 旧的车位/车辆标注
- **特征块（自由多边形）** —— 点若干点、右键闭合
- **HSV 标定（点一下箭头）** —— 点箭头 → 按 S，把颜色/几何/三阈值结果写进
  `vision/data/hsv_marker/calibration.jsonl`（**这个目录被 gitignore**，属于现场数据）

---

## 5. 依赖

| 用途 | 需要 |
|---|---|
| 固件 | ESP-IDF v5.5 |
| 视觉 | Python 3 + numpy + OpenCV（`start.sh` 默认用 `PY_VISION`，可用环境变量覆盖）|
| 单测 | `pytest` |
| RFID/地磁测试工具 | 纯标准库 + OpenCV（`tools/rfid/*.py` 只用 termios/select，不需要 pyserial）|
| PDF 资料阅读（可选）| `pip install pypdf` |

`start.sh` 里的默认解释器路径是作者机器的（`/usr/local/Caskroom/miniconda/.../vision-dev/bin/python`），
换机器时用 `PY_VISION=/path/to/python` 覆盖。

---

## 6. 常见故障与对应现象

| 现象 | 原因 | 处理 |
|---|---|---|
| 48 个车位全 `EMPTY`，`changed_ratio` 全 0 | 没有参考图，或参考图里有车 | 清空车位后按 `r` 重拍 |
| 页面卡住不动、`state.json` 不刷新 | `serve`/`vision` 没起来 | `./start.sh status`，看 `run/*.log` |
| 相机初始化失败 → 板子重启 | 有别的驱动占用了 SCCB 的 I2C 控制器 | SCCB 在 **控制器 1**，地磁必须用**控制器 0** |
| 地磁读到的 |B| 差 20 倍 | 量程判断错误 | QMC5883P 的量程要按实测反推（25–65 µT 校验）|
| RFID 突然只回响应、不出标签 | 之前发过"连续盘点"命令，模块状态被改且**跨 ESP32 重启保留** | **给模块断电**，只重启 ESP32 无效 |
| RFID 全部 select 失败 | 选通帧漏了校验字节 | 帧长必须 9 字节：`BB 00 1B 00 02 01 <n> <sum> 7E` |

---

## 7. 硬件接线

| 外设 | 参考 |
|---|---|
| 相机 / LCD / 灯带 / 道闸 | `README.md` §17（含舵机供电实测）|
| 地磁（TCA9548A + 4× QMC5883P）| `docs/geomagnetic.md` |
| RFID（4 天线 UHF）| `docs/rfid.md` §8 |

**GPIO 占用**（源码与 sdkconfig 核对过）：相机 4–18、LCD 19/20/21/38/45/47、灯带 14、红外 39、
舵机 40、地磁 41/42（I2C0）、控制台 43/44（UART0）、RFID 1/2（UART1）、flash 26–32、
Octal PSRAM 33–37。真正空闲：0、3、46、48（其中 0/3/46 是 strapping 脚）。

⚠️ **外部 JTAG 不可用**（39/41/40/42 全被占用），**原生 USB 不可用**（19/20 被 LCD 占用）
→ **43/44 是唯一的烧录与调试通道**，任何新外设都不能占这两个脚。
