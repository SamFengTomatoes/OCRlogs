#!/usr/bin/env python3
"""
K8s 容器内进程退出/被杀监控脚本

监控维度:
  1. 进程退出检测 — 实时发现目标进程消失，捕获最后状态
  2. Cgroup OOM 监控 — 监控 cgroup v1/v2 的 OOM kill 计数器变化
  3. 内核日志监听 — 读取 /dev/kmsg 捕获 OOM killer / segfault 等内核消息
  4. 系统内存监控 — 记录内存压力变化趋势
  5. 进程信号监控 — 轮询 /proc/<pid>/status 捕获待处理信号
  6. Wrapper 模式 — 以子进程方式启动目标程序，获取精确退出码

用法:
  # 模式1: 监控已有进程（按名称匹配）
  python3 process_kill_monitor.py --pattern "EngineCore|Worker|vllm"

  # 模式2: 以子进程方式启动（可获取精确退出码和信号）
  python3 process_kill_monitor.py --exec "vllm serve /model --port 8000"

  # 模式3: 监控指定 PID
  python3 process_kill_monitor.py --pids 1234,5678

  # 模式4: 监控 + 自动 dump 进程 /proc 信息
  python3 process_kill_monitor.py --pattern "EngineCore" --dump-proc
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── 常量 ──────────────────────────────────────────────────────────────────

LOG_DIR = Path(os.environ.get("MONITOR_LOG_DIR", "/opt/parampkg/proc_monitor"))
LOG_FILE = LOG_DIR / "monitor.log"
SNAPSHOT_DIR = LOG_DIR / "snapshots"

# 信号名称映射
SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL", 5: "SIGTRAP",
    6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL", 10: "SIGUSR1",
    11: "SIGSEGV", 12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
    16: "SIGSTKFLT", 17: "SIGCHLD", 18: "SIGCONT", 19: "SIGSTOP", 20: "SIGTSTP",
    21: "SIGTTIN", 22: "SIGTTOU", 23: "SIGURG", 24: "SIGXCPU", 25: "SIGXFSZ",
    26: "SIGVTALRM", 27: "SIGPROF", 28: "SIGWINCH", 29: "SIGIO", 30: "SIGPWR",
    31: "SIGSYS",
}

# libc 用于读取进程名
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


# ─── 日志工具 ──────────────────────────────────────────────────────────────

_log_lock = threading.Lock()

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"{ts} [{level:5s}] {msg}"
    with _log_lock:
        print(line, flush=True)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_json(data: dict, tag: str = "EVENT"):
    data["timestamp"] = datetime.now().isoformat()
    data["tag"] = tag
    log(json.dumps(data, ensure_ascii=False), "EVENT")


# ─── Cgroup 内存监控 ──────────────────────────────────────────────────────

class CgroupMemoryMonitor:
    """监控 cgroup v1 / v2 的内存事件，特别是 OOM kill。"""

    def __init__(self):
        self.version: int = 0        # 1 或 2
        self.base_path: str = ""
        self.last_oom_kill: int = 0
        self.last_oom: int = 0
        self._init_cgroup()

    def _init_cgroup(self):
        # cgroup v2: /sys/fs/cgroup/memory.events
        v2_events = Path("/sys/fs/cgroup/memory.events")
        # cgroup v1: /sys/fs/cgroup/memory/memory.oom_control
        v1_oom = Path("/sys/fs/cgroup/memory/memory.oom_control")

        # 也检查当前进程自身的 cgroup（k8s 容器可能使用嵌套 cgroup）
        self_cgroup_v2 = self._read_self_cgroup()

        if v2_events.exists():
            self.version = 2
            self.base_path = "/sys/fs/cgroup"
            log(f"Cgroup v2 detected, base={self.base_path}")
        elif self_cgroup_v2:
            # 尝试使用进程自身 cgroup 路径
            candidate = Path(self_cgroup_v2) / "memory.events"
            if candidate.exists():
                self.version = 2
                self.base_path = self_cgroup_v2
                log(f"Cgroup v2 (self), base={self.base_path}")
        elif v1_oom.exists():
            self.version = 1
            self.base_path = "/sys/fs/cgroup/memory"
            log(f"Cgroup v1 detected, base={self.base_path}")
        else:
            # 尝试通过 /proc/self/cgroup 找到实际路径
            self.base_path = self._find_cgroup_path()
            if self.base_path:
                if Path(self.base_path, "memory.events").exists():
                    self.version = 2
                elif Path(self.base_path, "memory.oom_control").exists():
                    self.version = 1
                log(f"Cgroup v{self.version} (auto), base={self.base_path}")
            else:
                log("WARNING: 无法检测 cgroup 版本，OOM 监控将不可用", "WARN")

        # 读取初始值
        self.last_oom_kill = self._get_oom_kill_count()
        self.last_oom = self._get_oom_count()
        log(f"初始 OOM kill 计数: {self.last_oom_kill}, OOM 计数: {self.last_oom}")

    def _read_self_cgroup(self) -> str:
        try:
            with open("/proc/self/cgroup") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) >= 3:
                        controller, path = parts[1], parts[2]
                        if controller == "memory" or controller == "":
                            return f"/sys/fs/cgroup{path}"
        except Exception:
            pass
        return ""

    def _find_cgroup_path(self) -> str:
        try:
            with open("/proc/self/mountinfo") as f:
                for line in f:
                    parts = line.split()
                    mount_point = parts[4]
                    fs_type = parts[-2] if len(parts) > 2 else ""
                    if "cgroup" in fs_type:
                        if Path(mount_point, "memory.events").exists():
                            return mount_point
                        if Path(mount_point, "memory.oom_control").exists():
                            return mount_point
        except Exception:
            pass
        return ""

    def _get_oom_kill_count(self) -> int:
        if self.version == 2:
            return self._read_v2_counter("oom_kill")
        elif self.version == 1:
            return self._read_v1_counter("oom_kill")
        return 0

    def _get_oom_count(self) -> int:
        if self.version == 2:
            return self._read_v2_counter("oom")
        elif self.version == 1:
            return self._read_v1_counter("under_oom")
        return 0

    def _read_v2_counter(self, key: str) -> int:
        try:
            with open(Path(self.base_path, "memory.events")) as f:
                for line in f:
                    k, v = line.strip().split()
                    if k == key:
                        return int(v)
        except Exception:
            pass
        return 0

    def _read_v1_counter(self, key: str) -> int:
        try:
            with open(Path(self.base_path, "memory.oom_control")) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[0] == key:
                        return int(parts[1])
        except Exception:
            pass
        return 0

    def get_memory_info(self) -> dict:
        info = {"cgroup_version": self.version, "base_path": self.base_path}
        if self.version == 2:
            try:
                with open(Path(self.base_path, "memory.current")) as f:
                    info["memory_current"] = int(f.read().strip())
                with open(Path(self.base_path, "memory.max")) as f:
                    val = f.read().strip()
                    info["memory_max"] = int(val) if val != "max" else -1
                with open(Path(self.base_path, "memory.events")) as f:
                    for line in f:
                        k, v = line.strip().split()
                        info[f"events_{k}"] = int(v)
            except Exception as e:
                info["error"] = str(e)
        elif self.version == 1:
            try:
                with open(Path(self.base_path, "memory.usage_in_bytes")) as f:
                    info["memory_current"] = int(f.read().strip())
                with open(Path(self.base_path, "memory.limit_in_bytes")) as f:
                    info["memory_max"] = int(f.read().strip())
            except Exception as e:
                info["error"] = str(e)
        return info

    def check_oom_delta(self) -> Optional[dict]:
        current_kill = self._get_oom_kill_count()
        current_oom = self._get_oom_count()
        if current_kill != self.last_oom_kill or current_oom != self.last_oom:
            delta = {
                "oom_kill_before": self.last_oom_kill,
                "oom_kill_after": current_kill,
                "oom_kill_delta": current_kill - self.last_oom_kill,
                "oom_before": self.last_oom,
                "oom_after": current_oom,
                "oom_delta": current_oom - self.last_oom,
                "memory_info": self.get_memory_info(),
            }
            self.last_oom_kill = current_kill
            self.last_oom = current_oom
            return delta
        return None

    def monitor_loop(self, stop_event: threading.Event, interval: float = 0.5):
        while not stop_event.wait(interval):
            delta = self.check_oom_delta()
            if delta:
                log_json(delta, "CGROUP_OOM_EVENT")
                log(f"!!! 检测到 cgroup OOM 事件: kill +{delta['oom_kill_delta']}, "
                    f"oom +{delta['oom_delta']}", "ALERT")


# ─── 内核日志监听 (/dev/kmsg) ─────────────────────────────────────────────

class KmsgMonitor:
    """
    读取 /dev/kmsg 实时捕获内核消息。
    需要读取权限（容器可能需要 securityContext.privileged 或 CAP_SYS_ADMIN）。
    """

    def __init__(self):
        self.enabled = False
        try:
            # 测试 /dev/kmsg 是否可读
            with open("/dev/kmsg", "rb") as f:
                pass
            self.enabled = True
            log("/dev/kmsg 可读，内核日志监听已启用")
        except PermissionError:
            log("/dev/kmsg 权限不足，内核日志监听不可用。"
                " 需要 CAP_SYS_ADMIN 或 privileged 容器", "WARN")
        except Exception as e:
            log(f"/dev/kmsg 不可用: {e}", "WARN")

    def monitor_loop(self, stop_event: threading.Event):
        if not self.enabled:
            return
        try:
            with open("/dev/kmsg", "rb") as f:
                while not stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    msg = line.decode("utf-8", errors="replace")
                    # 过滤关键消息
                    keywords = ["oom", "kill", "Killed process", "segfault",
                                "SIGSEGV", "SIGABRT", "out of memory",
                                "cgroup", "OOM", "panic"]
                    if any(kw in msg for kw in keywords):
                        log_json({
                            "kernel_message": msg.strip(),
                            "source": "/dev/kmsg",
                        }, "KMSG_ALERT")
                        log(f"!!! 内核消息: {msg.strip()[:200]}", "ALERT")
        except Exception as e:
            log(f"kmsg 监听异常: {e}", "ERROR")


# ─── 进程信息采集 ──────────────────────────────────────────────────────────

def read_proc_status(pid: int) -> dict:
    info = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    info[key] = val
    except Exception:
        pass
    return info


def read_proc_stat(pid: int) -> dict:
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read().split()
        return {
            "comm": data[1].strip("()"),
            "state": data[2],
            "ppid": int(data[3]),
            "vsize": int(data[22]),
            "rss": int(data[23]) * 4096,
        }
    except Exception:
        return {}


def get_oom_score(pid: int) -> dict:
    info = {}
    for key in ("oom_score", "oom_score_adj"):
        try:
            with open(f"/proc/{pid}/{key}") as f:
                info[key] = int(f.read().strip())
        except Exception:
            pass
    return info


def parse_sig_fields(val: str) -> list:
    """解析 /proc/<pid>/status 中的 SigPnd / ShdPnd 等十六进制信号位图。"""
    try:
        mask = int(val.split()[0], 16)
        return [SIGNAL_NAMES.get(i, f"SIG{i}") for i in range(1, 32) if mask & (1 << (i - 1))]
    except Exception:
        return []


def collect_proc_snapshot(pid: int) -> dict:
    """采集进程的完整 /proc 快照。"""
    snap = {"pid": pid, "timestamp": datetime.now().isoformat()}
    snap.update(read_proc_status(pid))
    snap.update(read_proc_stat(pid))
    snap.update(get_oom_score(pid))

    # 解析信号字段
    for sig_field in ("SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt"):
        if sig_field in snap:
            snap[f"{sig_field}_parsed"] = parse_sig_fields(snap[sig_field])

    # 内存详情
    try:
        with open(f"/proc/{pid}/statm") as f:
            statm = f.read().split()
            snap["statm_size"] = int(statm[0]) * 4096
            snap["statm_rss"] = int(statm[1]) * 4096
    except Exception:
        pass

    # 系统内存
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = parts[1].strip()
            snap["meminfo"] = meminfo
    except Exception:
        pass

    return snap


def save_snapshot(pid: int, label: str, data: dict):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = SNAPSHOT_DIR / f"{ts}_{pid}_{label}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log(f"快照已保存: {filename}")


# ─── 系统内存监控 ──────────────────────────────────────────────────────────

class SystemMemoryMonitor:
    def __init__(self):
        self.last_meminfo = {}
        self.last_pressure = {}

    def read_meminfo(self) -> dict:
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        info[key] = int(val)
        except Exception:
            pass
        return info

    def read_pressure(self) -> dict:
        info = {}
        for p in ("/proc/pressure/memory", "/proc/pressure/cpu", "/proc/pressure/io"):
            try:
                with open(p) as f:
                    name = Path(p).name
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            kind = parts[0]
                            vals = {}
                            for kv in parts[1:]:
                                if "=" in kv:
                                    k, v = kv.split("=")
                                    vals[k] = float(v)
                            info[f"{name}_{kind}"] = vals
            except Exception:
                pass
        return info

    def monitor_loop(self, stop_event: threading.Event, interval: float = 5.0):
        while not stop_event.wait(interval):
            meminfo = self.read_meminfo()
            pressure = self.read_pressure()

            # 检测内存压力变化
            alerts = []
            if pressure:
                for key, vals in pressure.items():
                    if isinstance(vals, dict):
                        if vals.get("avg10", 0) > 50:
                            alerts.append(f"{key} avg10={vals['avg10']}")

            # 记录可用内存低于阈值
            mem_available = meminfo.get("MemAvailable", 0)
            mem_total = meminfo.get("MemTotal", 1)
            mem_pct = (mem_available / mem_total * 100) if mem_total else 0

            if mem_pct < 10 or alerts:
                log_json({
                    "mem_available_mb": mem_available // 1024,
                    "mem_total_mb": mem_total // 1024,
                    "mem_available_pct": round(mem_pct, 2),
                    "pressure": pressure,
                    "alerts": alerts,
                }, "MEM_PRESSURE")
                if alerts:
                    log(f"!!! 内存压力告警: {', '.join(alerts)}", "ALERT")

            self.last_meminfo = meminfo
            self.last_pressure = pressure


# ─── 进程监控核心 ──────────────────────────────────────────────────────────

class ProcessMonitor:
    """
    监控目标进程，检测退出事件并采集退出原因。

    工作原理:
      1. 定期扫描匹配名称的进程，维护 {pid: last_snapshot} 映射
      2. 当发现进程消失时，立即采集上下文信息判断退出原因
      3. 在进程存活期间定期保存快照，用于退出后分析
    """

    def __init__(self, pattern: str = "", pids: list = None,
                 interval: float = 0.5, dump_proc: bool = False):
        self.pattern = re.compile(pattern) if pattern else None
        self.pids = set(pids or [])
        self.interval = interval
        self.dump_proc = dump_proc
        self.tracked: dict = {}           # {pid: last_snapshot}
        self.cg_monitor: Optional[CgroupMemoryMonitor] = None
        self._stop = threading.Event()

    def find_processes(self) -> dict:
        """扫描 /proc 找到匹配的进程。"""
        found = {}
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                comm = self._read_comm(pid)
                if not comm:
                    continue
                if self.pattern and self.pattern.search(comm):
                    found[pid] = comm
                elif pid in self.pids:
                    found[pid] = comm
        except Exception:
            pass
        return found

    @staticmethod
    def _read_comm(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/comm") as f:
                return f.read().strip()
        except Exception:
            return ""

    def monitor_loop(self):
        log(f"进程监控启动: pattern={self.pattern.pattern if self.pattern else None}, "
            f"pids={self.pids}, interval={self.interval}s")

        snapshot_counter = 0
        while not self._stop.is_set():
            current = self.find_processes()

            # 检测新进程
            for pid, comm in current.items():
                if pid not in self.tracked:
                    snap = collect_proc_snapshot(pid)
                    self.tracked[pid] = snap
                    log_json({
                        "pid": pid, "comm": comm,
                        "ppid": snap.get("PPid"),
                        "state": snap.get("State"),
                        "oom_score": snap.get("oom_score"),
                        "vsize_mb": snap.get("vsize", 0) // (1024 * 1024),
                        "rss_mb": snap.get("rss", 0) // (1024 * 1024),
                    }, "PROC_START")
                    log(f">> 进程出现: PID={pid} comm={comm} "
                        f"RSS={snap.get('rss', 0) // (1024*1024)}MB")

            # 检测退出进程
            exited = set(self.tracked.keys()) - set(current.keys())
            for pid in exited:
                last_snap = self.tracked.pop(pid)
                self._handle_exit(pid, last_snap)

            # 定期快照
            snapshot_counter += 1
            snapshot_interval = max(1, int(10 / self.interval))
            if snapshot_counter % snapshot_interval == 0 and self.dump_proc:
                for pid in self.tracked:
                    snap = collect_proc_snapshot(pid)
                    save_snapshot(pid, "periodic", snap)
                    self.tracked[pid] = snap

            self._stop.wait(self.interval)

    def _handle_exit(self, pid: int, last_snap: dict):
        comm = last_snap.get("comm", last_snap.get("Name", "unknown"))
        log(f"<< 进程消失: PID={pid} comm={comm}", "ALERT")

        # 立即检查 cgroup OOM
        oom_delta = None
        if self.cg_monitor:
            oom_delta = self.cg_monitor.check_oom_delta()

        # 采集系统内存状态
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
        except Exception:
            meminfo = {}

        # 读取内存压力
        pressure = {}
        try:
            with open("/proc/pressure/memory") as f:
                pressure["memory"] = f.read().strip()
        except Exception:
            pass

        # 尝试读取 dmesg 最后几行（可能有权限）
        dmesg_tail = []
        try:
            result = subprocess.run(
                ["dmesg", "--ctime", "--nopager"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                dmesg_tail = lines[-50:] if len(lines) > 50 else lines
        except Exception:
            pass

        # 检查 /proc/<pid> 是否还存在（可能是 zombie）
        zombie_info = {}
        proc_status = read_proc_status(pid)
        if proc_status:
            zombie_info = {
                "still_in_proc": True,
                "state": proc_status.get("State"),
                "status": proc_status,
            }

        # 构建退出事件
        exit_event = {
            "pid": pid,
            "comm": comm,
            "exit_time": datetime.now().isoformat(),
            "last_known_state": {
                "State": last_snap.get("State"),
                "VmRSS": last_snap.get("VmRSS"),
                "VmSize": last_snap.get("VmSize"),
                "oom_score": last_snap.get("oom_score"),
                "SigPnd_parsed": last_snap.get("SigPnd_parsed", []),
                "ShdPnd_parsed": last_snap.get("ShdPnd_parsed", []),
            },
            "cgroup_oom_delta": oom_delta,
            "system_meminfo": meminfo,
            "memory_pressure": pressure,
            "zombie_info": zombie_info,
            "dmesg_tail": dmesg_tail,
        }

        # 退出原因推断
        reason = self._infer_exit_reason(exit_event)
        exit_event["inferred_reason"] = reason
        exit_event["last_snapshot"] = last_snap

        log_json(exit_event, "PROC_EXIT")

        reason_str = json.dumps(reason, ensure_ascii=False, indent=2)
        log(f"!!! 进程退出分析 PID={pid} comm={comm}:", "ALERT")
        for line in reason_str.split("\n"):
            log(f"    {line}", "ALERT")

        # 保存完整快照
        save_snapshot(pid, "exit", exit_event)

    def _infer_exit_reason(self, event: dict) -> dict:
        """根据上下文推断退出原因。"""
        reasons = []
        confidence = "LOW"

        # 检查 cgroup OOM
        oom_delta = event.get("cgroup_oom_delta")
        if oom_delta and oom_delta.get("oom_kill_delta", 0) > 0:
            reasons.append("CGROUP_OOM_KILL")
            confidence = "HIGH"

        # 检查系统内存
        meminfo = event.get("system_meminfo", {})
        mem_available_str = meminfo.get("MemAvailable", "0")
        try:
            mem_available = int(mem_available_str.split()[0])
        except (ValueError, IndexError):
            mem_available = 0
        if mem_available and mem_available < 100 * 1024:  # < 100MB
            reasons.append("SYSTEM_MEMORY_NEAR_EXHAUSTION")
            if confidence == "LOW":
                confidence = "MEDIUM"

        # 检查最后的信号
        last_state = event.get("last_known_state", {})
        sig_pnd = last_state.get("SigPnd_parsed", [])
        shd_pnd = last_state.get("ShdPnd_parsed", [])
        if sig_pnd or shd_pnd:
            reasons.append(f"PENDING_SIGNALS: sig={sig_pnd}, shd={shd_pnd}")

        # 检查 oom_score
        oom_score = last_state.get("oom_score")
        if oom_score is not None and oom_score > 500:
            reasons.append(f"HIGH_OOM_SCORE: {oom_score}")
            if "SYSTEM_MEMORY_NEAR_EXHAUSTION" in reasons:
                confidence = "MEDIUM"

        # 检查 dmesg
        dmesg_tail = event.get("dmesg_tail", [])
        for line in dmesg_tail:
            line_lower = line.lower()
            if "killed process" in line_lower or "out of memory" in line_lower:
                reasons.append(f"DMESG_OOM: {line.strip()[:200]}")
                confidence = "HIGH"
                break
            if "segfault" in line_lower:
                reasons.append(f"DMESG_SEGFAULT: {line.strip()[:200]}")
                confidence = "HIGH"
                break

        if not reasons:
            reasons.append("UNKNOWN — 可能是外部SIGKILL、原生崩溃或应用主动终止")

        return {
            "reasons": reasons,
            "confidence": confidence,
            "note": "若需精确退出码，请使用 --exec 模式启动目标进程",
        }

    def stop(self):
        self._stop.set()


# ─── Wrapper 模式（子进程精确退出码） ─────────────────────────────────────

def run_wrapper(command: str, cg_monitor: CgroupMemoryMonitor,
                sys_mem_monitor: SystemMemoryMonitor,
                dump_proc: bool = False):
    """
    以子进程方式启动目标命令，可获取精确退出码和信号。
    同时监控所有子进程。
    """
    log(f"Wrapper 模式启动: {command}")

    stop_event = threading.Event()

    # 启动后台监控线程
    threads = []
    if cg_monitor:
        t = threading.Thread(target=cg_monitor.monitor_loop,
                             args=(stop_event,), daemon=True)
        t.start()
        threads.append(t)

    kmsg = KmsgMonitor()
    t = threading.Thread(target=kmsg.monitor_loop, args=(stop_event,),
                         daemon=True)
    t.start()
    threads.append(t)

    t = threading.Thread(target=sys_mem_monitor.monitor_loop,
                         args=(stop_event,), daemon=True)
    t.start()
    threads.append(t)

    # 启动目标进程
    import shlex
    args = shlex.split(command)
    proc = subprocess.Popen(args)

    log(f"子进程已启动: PID={proc.pid}")

    # 监控子进程及其所有后代
    tracked_children = {}

    def monitor_children():
        while not stop_event.is_set():
            try:
                # 扫描 proc.pid 的所有后代进程
                descendants = _get_descendants(proc.pid)
                for dpid, dcomm in descendants.items():
                    if dpid not in tracked_children:
                        snap = collect_proc_snapshot(dpid)
                        tracked_children[dpid] = snap
                        log_json({"pid": dpid, "comm": dcomm,
                                  "ppid": proc.pid}, "CHILD_START")
                        log(f">> 子进程出现: PID={dpid} comm={dcomm}")

                # 检测子进程退出
                exited = set(tracked_children.keys()) - set(descendants.keys())
                for dpid in exited:
                    last = tracked_children.pop(dpid)
                    log_json({
                        "pid": dpid,
                        "comm": last.get("Name", "?"),
                        "last_state": last.get("State"),
                        "last_rss": last.get("VmRSS"),
                    }, "CHILD_EXIT")
                    log(f"<< 子进程消失: PID={dpid}", "WARN")

                    # 检查 OOM
                    delta = cg_monitor.check_oom_delta() if cg_monitor else None
                    if delta and delta.get("oom_kill_delta", 0) > 0:
                        log(f"!!! 子进程 PID={dpid} 可能被 cgroup OOM 杀死!", "ALERT")

            except Exception as e:
                log(f"子进程监控异常: {e}", "ERROR")

            stop_event.wait(1.0)

    child_thread = threading.Thread(target=monitor_children, daemon=True)
    child_thread.start()

    # 等待主进程退出
    ret = proc.wait()

    stop_event.set()

    # 获取退出信息
    exit_code = ret
    signal_num = None
    if ret < 0:
        signal_num = -ret
        signal_name = SIGNAL_NAMES.get(signal_num, f"SIG{signal_num}")
    elif ret > 128:
        signal_num = ret - 128
        signal_name = SIGNAL_NAMES.get(signal_num, f"SIG{signal_num}")
    else:
        signal_name = None

    exit_info = {
        "pid": proc.pid,
        "exit_code": exit_code,
        "signal_num": signal_num,
        "signal_name": signal_name,
        "exit_time": datetime.now().isoformat(),
        "cgroup_oom_delta": cg_monitor.check_oom_delta() if cg_monitor else None,
        "system_meminfo": sys_mem_monitor.read_meminfo(),
        "memory_pressure": sys_mem_monitor.read_pressure(),
    }

    log_json(exit_info, "WRAPPER_EXIT")

    if signal_name:
        log(f"!!! 主进程被信号杀死: PID={proc.pid} signal={signal_name} "
            f"(code={exit_code})", "ALERT")
        if signal_name == "SIGKILL":
            # 进一步分析 SIGKILL 的来源
            oom_delta = exit_info.get("cgroup_oom_delta")
            if oom_delta and oom_delta.get("oom_kill_delta", 0) > 0:
                log("    -> 原因推断: CGROUP OOM KILL (高可信)", "ALERT")
            else:
                log("    -> 原因推断: 外部SIGKILL（可能是系统OOM killer、"
                    "k8s eviction、或应用主动kill）", "ALERT")
        elif signal_name == "SIGSEGV":
            log("    -> 原因推断: 原生代码段错误（native crash）", "ALERT")
        elif signal_name == "SIGABRT":
            log("    -> 原因推断: abort()调用（可能是NCCL/C++断言失败）", "ALERT")
        elif signal_name == "SIGTERM":
            log("    -> 原因推断: 外部优雅终止请求", "ALERT")
    elif exit_code == 0:
        log(f"主进程正常退出: PID={proc.pid} code=0")
    else:
        log(f"主进程异常退出: PID={proc.pid} code={exit_code}", "WARN")

    save_snapshot(proc.pid, "wrapper_exit", exit_info)

    return exit_code


def _get_descendants(root_pid: int) -> dict:
    """获取某 PID 的所有后代进程。"""
    result = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == root_pid:
                continue
            stat = read_proc_stat(pid)
            ppid = stat.get("ppid", -1)
            if ppid == root_pid or ppid in result:
                comm = stat.get("comm", "")
                result[pid] = comm
    except Exception:
        pass
    return result


# ─── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="K8s 容器内进程退出/被杀监控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pattern", "-p", type=str,
                       help="进程名称匹配正则表达式（如 'EngineCore|Worker'）")
    group.add_argument("--exec", "-e", type=str,
                       help="以子进程方式启动命令（可获取精确退出码）")
    group.add_argument("--pids", type=str,
                       help="监控指定 PID 列表（逗号分隔，如 1234,5678）")

    parser.add_argument("--interval", type=float, default=0.5,
                        help="进程扫描间隔秒数（默认 0.5）")
    parser.add_argument("--dump-proc", action="store_true",
                        help="定期 dump 进程 /proc 信息到快照目录")
    parser.add_argument("--log-dir", type=str, default="",
                        help="日志输出目录（默认 /opt/parampkg/proc_monitor）")

    args = parser.parse_args()

    if args.log_dir:
        global LOG_DIR, LOG_FILE, SNAPSHOT_DIR
        LOG_DIR = Path(args.log_dir)
        LOG_FILE = LOG_DIR / "monitor.log"
        SNAPSHOT_DIR = LOG_DIR / "snapshots"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("K8s 容器进程退出监控脚本 启动")
    log(f"  日志目录: {LOG_DIR}")
    log(f"  快照目录: {SNAPSHOT_DIR}")
    log(f"  Python: {sys.version}")
    log(f"  PID: {os.getpid()}")
    log(f"  参数: {vars(args)}")
    log("=" * 70)

    # 初始化 cgroup 监控
    cg_monitor = CgroupMemoryMonitor()

    # 初始化系统内存监控
    sys_mem_monitor = SystemMemoryMonitor()

    # 初始化 kmsg 监控
    kmsg_monitor = KmsgMonitor()

    # ── Wrapper 模式 ──
    if args.exec:
        exit_code = run_wrapper(args.exec, cg_monitor, sys_mem_monitor,
                                args.dump_proc)
        sys.exit(exit_code)

    # ── 监控模式 ──
    stop_event = threading.Event()

    # 启动 cgroup OOM 监控线程
    if cg_monitor.version > 0:
        t = threading.Thread(target=cg_monitor.monitor_loop,
                             args=(stop_event, args.interval), daemon=True)
        t.start()

    # 启动 kmsg 监控线程
    t = threading.Thread(target=kmsg_monitor.monitor_loop,
                         args=(stop_event,), daemon=True)
    t.start()

    # 启动系统内存监控线程
    t = threading.Thread(target=sys_mem_monitor.monitor_loop,
                         args=(stop_event, 5.0), daemon=True)
    t.start()

    # 启动进程监控
    pids = None
    if args.pids:
        pids = [int(p) for p in args.pids.split(",")]

    proc_monitor = ProcessMonitor(
        pattern=args.pattern,
        pids=pids,
        interval=args.interval,
        dump_proc=args.dump_proc,
    )
    proc_monitor.cg_monitor = cg_monitor

    def signal_handler(signum, frame):
        log(f"监控脚本收到信号 {SIGNAL_NAMES.get(signum, signum)}，正在停止...")
        stop_event.set()
        proc_monitor.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        proc_monitor.monitor_loop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        proc_monitor.stop()
        log("监控脚本已停止")


if __name__ == "__main__":
    main()
