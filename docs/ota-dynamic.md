# OTA 动态寻址、自动恢复与发布认证（0.2.76）

## 寻址

`ota.c` 复用 `discovery_url()` 与 `discovery_host()`，不另建发现框架。
beacon 每 2 秒广播一次，记录 15 秒有效。OTA 每 30 秒检查一次，也可由
`POST /api/ota/check` 立即唤醒。

```text
发现的 PC + CONFIG_OTA_MANIFEST_URL 的端口/路径
  → http://<当前PC>:8000/manifest.json
```

未发现 PC 或 Wi-Fi 断开时不发 HTTP，也不使用过期地址。

## 状态

设备首页同时显示实际 manifest URL、状态和最近结果：

- `PC_NOT_FOUND`：无网络或没有有效发现记录；
- `PC_DISCOVERED`：已发现后台，尚未确认 OTA；
- `OTA_UNAVAILABLE`：manifest 或固件下载不可用；
- `OTA_AVAILABLE`：签名 manifest 已取得；
- `OTA_VERIFY_FAILED`：签名、来源、SHA-256 或镜像校验失败。

OTA 服务恢复、PC 重新上线或 IP 变化后，下一轮 30 秒检查自动恢复。

## 发布者认证

Discovery 只负责寻址，不授权发布固件。manifest 同时包含：

- `sha256`：固件完整性；
- `signature_algorithm = HMAC-SHA256`；
- `signature`：发布者对规范化 manifest 的认证。

设备先验证 HMAC，再进入原版本比较、同源下载、SHA-256、A/B 写入与回滚流程。
缺签名、错误签名、跨源固件 URL 和错误 SHA-256 均拒绝安装。

本机密钥：`~/.config/parking-lot/ota.key`（权限 600）。
固件密钥：本地 `sdkconfig` 的 `CONFIG_OTA_PUBLISHER_KEY`，不要提交 Git。
`PARKING_OTA_KEY_FILE` 可覆盖发布工具使用的密钥文件。

发布：

```bash
./tools/ota-server/publish.sh <version>
```

## 换 PC / 网络

1. 新 PC 启动 `./start.sh`（beacon 与 OTA 默认启动）；
2. 将同一 `ota.key` 安全复制到新 PC 的相同路径；
3. 在新 PC 发布 manifest/固件；
4. ESP32 无需改地址或重新烧录，下一轮检查发现并验证新后台。

未知 Wi-Fi 的密码仍需通过既有配网流程提供。

## 验证

`bash tools/tests/run.sh` 覆盖版本比较、URL host 改写、矩阵和签名/篡改测试。
`idf.py build` 验证完整固件。真机仍需分别验证正常升级、停/起 OTA、PC 更换、
错误签名、错误 SHA 和其它硬件无回退。
