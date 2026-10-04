# Thin OTA server

Two files over HTTP, and nothing else:

```
GET /manifest.json
GET /firmware/<version>/parking-lot-cam.bin
```

No accounts, no login, no database, no device management, no fleet
management, no dashboard, no telemetry, no cloud deployment, no firmware
metadata service. Python standard library only — there is no Flask/FastAPI
dependency to install.

---

## Run it

```bash
cd tools/ota-server
./run.sh                 # port 8000, all interfaces
./run.sh --port 9000     # or a different port
```

The startup banner prints the LAN URL to put into the firmware. Take the
`device ` line verbatim and paste it into the firmware's single OTA address:

```
cd ../../firmware/esp32-s3-cam
idf.py menuconfig      # Parking Lot Camera -> OTA manifest URL
```

(each launch prints one or more `device   : http://<lan-ip>:8000/manifest.json`
lines, one per local interface - pick the address the ESP32 can reach)

## Publish a firmware

```bash
cd firmware/esp32-s3-cam && idf.py build        # produces build/parking-lot-cam.bin
cd ../../tools/ota-server
./publish.sh 0.2.1
```

`publish.sh` copies the image to `firmware/0.2.1/parking-lot-cam.bin`, computes
its SHA-256 and writes `manifest.json`. Both are gitignored: they are release
artifacts, not source.

To publish a specific image instead of the default build output:

```bash
./publish.sh 0.2.1 /path/to/parking-lot-cam.bin
```

## Manifest schema

Deliberately small:

```json
{
  "project": "parking-lot-cam",
  "version": "0.2.1",
  "firmware_url": "/firmware/0.2.1/parking-lot-cam.bin",
  "sha256": "3b1f...e9",
  "size": 906112
}
```

| field | required | meaning |
|---|---|---|
| `project` | no | informational only; the device ignores it |
| `version` | **yes** | `major.minor.patch`, compared as integers |
| `firmware_url` | **yes** | absolute, or relative to the manifest URL |
| `sha256` | **yes** | lowercase hex digest of the binary |
| `size` | no | when present the device checks that it received exactly this many bytes |

## Verifying that a bad release is refused

`publish.sh` can emit a deliberately wrong digest, which is how the
"OTA failure must not break the running firmware" property is checked:

```bash
./publish.sh --sha256 0000000000000000000000000000000000000000000000000000000000000000 9.9.9
# device logs: "SHA256 mismatch: expected 0000... got <real>"
# device keeps running the old firmware
./publish.sh 0.2.1     # restore a good manifest
```

## HTTP or HTTPS?

The server speaks plain HTTP on the sandbox LAN, and the device always verifies
the image against the manifest's SHA-256 before committing it, so a corrupted or
truncated download can never be installed.

The firmware also accepts `https://` URLs and validates them against the ESP
x509 root bundle (no PKI work needed), so putting this behind a normal TLS
reverse proxy is the production path. Do **not** publish firmware over plain
HTTP to an untrusted network.
