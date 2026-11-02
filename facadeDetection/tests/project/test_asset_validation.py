"""项目资源校验的实效验证：文件修改时间未变时跳过全量哈希，变了必须重新哈希。

仅使用 tmp_path 下的普通文件，不创建真实项目目录或数据库。
"""
from __future__ import annotations

import hashlib
import os
import time
from types import SimpleNamespace

from services.dal.file_repo import FileRepo
import services.dal.file_repo as file_repo_mod


# ---------------------------------------------------------------------------
# validate_asset：mtime 未变跳过全量哈希，mtime 变化必须重新哈希
# ---------------------------------------------------------------------------
class TestValidateAssetMtimeShortcircuit:
    def test_unchanged_file_skips_sha256_changed_file_rehashes(
            self, tmp_path, monkeypatch):
        payload = os.urandom(16 * 1024 * 1024)  # 16MB，哈希耗时真实可测
        target = tmp_path / 'scan.ply'
        target.write_bytes(payload)
        real_sha = hashlib.sha256(payload).hexdigest()
        stat = target.stat()

        hash_calls = []
        real_sha256 = file_repo_mod._sha256
        def counting_sha256(path):
            t0 = time.perf_counter()
            digest = real_sha256(path)
            hash_calls.append((str(path), digest, time.perf_counter() - t0))
            return digest
        monkeypatch.setattr(file_repo_mod, '_sha256', counting_sha256)

        asset = SimpleNamespace(
            path=str(target), size_bytes=stat.st_size, sha256=real_sha,
            meta_json={'mtime_ns': stat.st_mtime_ns})

        # mtime 命中：一次哈希都不应发生
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert hash_calls == []

        # 无 mtime 记录：回退到真实哈希，且哈希值必须等于真实内容摘要
        asset.meta_json = {}
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert len(hash_calls) == 1
        assert hash_calls[0][1] == real_sha  # 机器真的读了 16MB 并算出正确摘要
        assert hash_calls[0][2] > 0
        assert asset.meta_json['mtime_ns'] == stat.st_mtime_ns  # 哈希后回写 mtime

        # 再次校验：mtime 已记录，恢复短路
        ok, _ = FileRepo.validate_asset(asset)
        assert ok and len(hash_calls) == 1

        # mtime 改变：必须重新哈希（防止短路掩盖真实修改）
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert len(hash_calls) == 2
        print(f'\n[perf] sha256 16MB = {hash_calls[0][2]*1e3:.2f}ms, '
              f'skipped_calls_saved=2')
