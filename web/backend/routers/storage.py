"""JSON 存储工具 — 原子写 + 容错读

问题背景:
  - 原先 router 直接 json.dump/open 目标文件, 并发请求或进程中断会截断
    strategy_config.json / watchlist.json; 且 _load_config 无 try/except,
    文件一旦损坏, 所有相关接口将持续 500.
  - 这里统一提供:
      atomic_write_json: 写临时文件 → fsync → os.replace (同文件系统原子替换)
      safe_load_json: 读取容错, 损坏/缺失时返回默认值而非抛异常
"""
import json
import os
import tempfile


def atomic_write_json(path: str, data) -> None:
    """原子写入 JSON: 先写临时文件并落盘, 再原子替换目标文件.

    即使写入途中进程被中断, 目标文件要么保持旧内容, 要么已是完整新内容,
    不会出现半截文件导致后续读取失败.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def safe_load_json(path: str, default):
    """容错读取 JSON: 文件缺失或损坏时返回 default, 不抛异常.

    避免目标文件损坏后, 所有依赖它的接口持续 500.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        # 文件损坏: 返回默认值并静默降级(调用方应记录日志)
        return default
