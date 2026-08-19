#!/usr/bin/env python3
"""
加密层：API Key 的认证加密存储与密钥配置读写。
依赖 storage.py 的路径常量；零第三方依赖。
"""
import base64
import hashlib
import hmac
import json
import secrets

from storage import CONFIG_FILE, SECRET_FILE


def _master_key() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(key)
    try:
        import os
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return key


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def encrypt_str(plaintext: str) -> str:
    if not plaintext:
        return ""
    master = _master_key()
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", master, salt, 120_000, dklen=64)
    enc_key, mac_key = dk[:32], dk[32:]
    data = plaintext.encode("utf-8")
    ct = bytes(a ^ b for a, b in zip(data, _keystream(enc_key, nonce, len(data))))
    tag = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(salt + nonce + tag + ct).decode("ascii")


def decrypt_str(token: str) -> str:
    if not token:
        return ""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception:
        return ""
    if len(raw) < 64:
        return ""
    salt, nonce, tag, ct = raw[:16], raw[16:32], raw[32:64], raw[64:]
    master = _master_key()
    dk = hashlib.pbkdf2_hmac("sha256", master, salt, 120_000, dklen=64)
    enc_key, mac_key = dk[:32], dk[32:]
    expect = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(expect, tag):
        return ""
    data = bytes(a ^ b for a, b in zip(ct, _keystream(enc_key, nonce, len(ct))))
    return data.decode("utf-8")


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[0] + "*" * (len(key) - 1)
    return key[:6] + "••••••" + key[-4:]


# ---------- 运行时密钥：用户自定义优先，否则用内置默认值 ----------
_saved_providers = set()
_runtime_keys = {}


def init_runtime_keys(defaults: dict) -> dict:
    """加载保存的密钥（覆盖 defaults），返回运行时密钥表。服务启动时调用一次。

    注意：原地更新 _runtime_keys（clear+update），保证 `from crypto import _runtime_keys`
    的导入方拿到的是同一字典对象。
    """
    keys = dict(defaults)
    for provider in keys:
        enc = _load_config().get(provider)
        if enc:
            val = decrypt_str(enc)
            if val:
                keys[provider] = val
                _saved_providers.add(provider)
    _runtime_keys.clear()
    _runtime_keys.update(keys)
    return _runtime_keys
