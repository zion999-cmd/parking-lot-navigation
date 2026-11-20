"""Publisher signature tests for dynamic OTA (stdlib only)."""

import hashlib
import hmac
import importlib.util
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "ota-server", "manifest_sign.py")
spec = importlib.util.spec_from_file_location("manifest_sign", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

KEY = b"test-only-publisher-key-with-at-least-32-bytes"
DOC = {"project": "parking-lot-cam", "version": "0.2.76",
       "firmware_url": "/firmware/0.2.76/parking-lot-cam.bin",
       "sha256": "ab" * 32, "size": 1408720}


def main():
    signed = module.sign(DOC, KEY)
    assert signed["signature_algorithm"] == "HMAC-SHA256"
    expected = hmac.new(KEY, module.canonical(DOC), hashlib.sha256).hexdigest()
    assert signed["signature"] == expected
    assert hmac.compare_digest(signed["signature"], expected)

    changed = dict(DOC, version="9.9.9")
    forged = module.sign(changed, KEY)
    assert forged["signature"] != signed["signature"], "version was not signed"
    wrong_key = module.sign(DOC, b"wrong-key-that-is-also-at-least-32-bytes")
    assert wrong_key["signature"] != signed["signature"], "wrong key was accepted"

    changed_url = dict(DOC, firmware_url="http://attacker/firmware.bin")
    assert module.sign(changed_url, KEY)["signature"] != signed["signature"]
    changed_sha = dict(DOC, sha256="cd" * 32)
    assert module.sign(changed_sha, KEY)["signature"] != signed["signature"]
    assert module.canonical(DOC).endswith(b"\n")
    print("ota manifest: 6 checks, 0 failures")


if __name__ == "__main__":
    main()
