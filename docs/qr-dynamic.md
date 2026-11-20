# 动态导航二维码（0.2.74 起；0.2.75 修正紧凑行布局）

## 数据流

```text
PC beacon（UDP 9999）
  → discovery.c 取得当前 PC 地址
  → qr_dynamic 任务 GET http://<PC>:8010/api/qr
  → 校验 URL 与矩阵
  → RAM 中的 qr_snapshot_t
  → display.c 复用原有 LCD 模块绘制
```

二维码 URL 由 PC 根据请求的 `Host` 生成，统一指向当前后台的 `/go`。
`qr_dynamic` 是独立低优先级任务；gate 和 display 任务均不执行 HTTP。

## 有效性与失效

只有以下检查全部通过，快照才有效：

- `schema == parking_qr/1`；
- `url` 精确等于数据来源主机的 `/go`，防止显示旧 PC 的矩阵；
- QR 版本尺寸为 21–45 且符合版本步进，行数、行长、0/1 内容正确；
- 每行未使用位为 0；
- 三个 7×7 定位图形完整。

以下来源变化或错误会立即使快照失效：PC 地址改变、连接失败、非 200、
空响应、错误 JSON、URL 不同源、错误矩阵。overlay 仍由原来的 boot/gate
计时器控制，但内容变为 `QR unavailable`，不显示旧地址。

## 诊断

- PC 数据：`GET http://<PC>:8010/api/qr`
- 设备状态：`GET http://<板子IP>/api/qr-status`
- 设备首页：QR 有效位、当前 URL、最后成功时间/错误原因

`qr-status` 字段：`valid`、`url`、`source_url`、`last_attempt_age_ms`、
`last_success_age_ms`、`error`。

## 测试

```bash
python vision/tests/test_qr.py   # PC-A/PC-B + OpenCV 实际解码 + 非法 Host
bash tools/tests/run.sh          # 原有测试 + 错误矩阵/填充位/定位图形
```

真机验收（不能用单元测试替代）：

1. PC-A 启动 `./start.sh`，触发 QR，手机扫码必须打开 PC-A 导航；
2. 停 PC-A 的 beacon/服务，在 PC-B 启动；观察 `qr-status` 切换到 PC-B；
3. 手机再次扫码必须打开 PC-B，期间不得继续显示 PC-A 旧码；
4. 停 serve，LCD 显示不可用且 `valid=false`；恢复 serve 后自动恢复；
5. 返回错误 JSON、错误尺寸或损坏矩阵，应拒绝并记录原因；
6. 对比道闸抬落杆、QR linger、视频、LED、RFID、地磁和红外行为，无回退。
