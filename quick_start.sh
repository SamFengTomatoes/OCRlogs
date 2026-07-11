#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 快速部署脚本：在 k8s 容器内启动进程监控
#
# 用法:
#   方式1: 直接在已有容器中运行
#     kubectl exec -it <pod> -- bash -c "$(cat quick_start.sh)"
#
#   方式2: 将脚本拷入容器后运行
#     kubectl cp process_kill_monitor.py <pod>:/tmp/
#     kubectl cp quick_start.sh <pod>:/tmp/
#     kubectl exec -it <pod> -- bash /tmp/quick_start.sh
# ──────────────────────────────────────────────────────────────────────────

set -e

MONITOR_DIR="/opt/parampkg/proc_monitor"
SCRIPT_NAME="process_kill_monitor.py"

mkdir -p "$MONITOR_DIR"

# 检查 Python3
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 未找到"
    exit 1
fi

# 检查 psutil（vLLM 容器通常已有）
python3 -c "import psutil" 2>/dev/null || pip3 install psutil -q 2>/dev/null || true

echo "============================================"
echo "  进程退出监控 - 快速启动"
echo "============================================"
echo "  监控目录: $MONITOR_DIR"
echo "  日志文件: $MONITOR_DIR/monitor.log"
echo "  快照目录: $MONITOR_DIR/snapshots/"
echo ""

# 显示当前匹配的进程
echo "[INFO] 当前匹配的进程:"
ps aux | grep -iE "EngineCore|Worker|vllm" | grep -v grep || echo "  (暂无匹配进程)"
echo ""

# 检查权限
echo "[INFO] 权限检查:"
if [ -r /dev/kmsg ]; then
    echo "  /dev/kmsg: 可读 (内核日志监听可用)"
else
    echo "  /dev/kmsg: 不可读 (需要 privileged 容器)"
fi

if [ -f /sys/fs/cgroup/memory.events ]; then
    echo "  cgroup v2: memory.events 可读"
    cat /sys/fs/cgroup/memory.events
elif [ -f /sys/fs/cgroup/memory/memory.oom_control ]; then
    echo "  cgroup v1: memory.oom_control 可读"
    cat /sys/fs/cgroup/memory/memory.oom_control
else
    echo "  cgroup: 无法确定版本"
fi
echo ""

echo "[INFO] 启动监控 (Ctrl+C 停止)..."
echo ""

# 启动监控
python3 "$SCRIPT_NAME" \
    --pattern "EngineCore|Worker|vllm" \
    --interval 0.5 \
    --dump-proc \
    --log-dir "$MONITOR_DIR"
