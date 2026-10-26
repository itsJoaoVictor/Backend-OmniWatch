import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import platform
import re
import threading
import time
from typing import Any, Dict, List, Optional
import psutil

logger = logging.getLogger("omniwatch.admin")

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEEDTEST_CACHE_FILE = os.path.join(CACHE_DIR, ".speedtest_cache.json")

# Regex para normalizar rotas (substituir UUIDs e inteiros por :id)
UUID_REGEX = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
NUMBER_REGEX = re.compile(r"/\d+(?=/|$)")

def normalize_path(path: str) -> str:
    cleaned = UUID_REGEX.sub(":id", path)
    cleaned = NUMBER_REGEX.sub("/:id", cleaned)
    return cleaned

class APIMetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_duration_ms = 0.0
        self.status_counts = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        
        # Deque com timestamps para cálculo de requisições por minuto (janela de 60s)
        self.recent_request_timestamps = deque()
        
        # Estatísticas por rota normalizada: {route: {count: int, total_duration: float, errors: int}}
        self.routes_stats: Dict[str, Dict[str, Any]] = {}
        
        # Deque com os últimos 30 erros
        self.recent_errors = deque(maxlen=30)

    def record(self, method: str, raw_path: str, status_code: int, duration_ms: float, client_ip: str):
        now = time.time()
        norm_path = normalize_path(raw_path)
        route_key = f"{method} {norm_path}"
        
        # Determina o grupo de status HTTP
        group = "5xx"
        if 200 <= status_code < 300:
            group = "2xx"
        elif 300 <= status_code < 400:
            group = "3xx"
        elif 400 <= status_code < 500:
            group = "4xx"

        is_error = status_code >= 400

        with self._lock:
            self.total_requests += 1
            self.total_duration_ms += duration_ms
            self.status_counts[group] = self.status_counts.get(group, 0) + 1

            self.recent_request_timestamps.append(now)

            if route_key not in self.routes_stats:
                self.routes_stats[route_key] = {
                    "method": method,
                    "path": norm_path,
                    "count": 0,
                    "total_duration_ms": 0.0,
                    "errors": 0,
                }
            
            r_stat = self.routes_stats[route_key]
            r_stat["count"] += 1
            r_stat["total_duration_ms"] += duration_ms
            if is_error:
                r_stat["errors"] += 1

            if is_error:
                self.recent_errors.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "method": method,
                    "path": raw_path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                })

    def get_summary(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            # Limpa timestamps mais antigos que 60 segundos
            while self.recent_request_timestamps and (now - self.recent_request_timestamps[0] > 60):
                self.recent_request_timestamps.popleft()
            
            rpm = len(self.recent_request_timestamps)
            rps = round(rpm / 60.0, 2)
            avg_latency = round(self.total_duration_ms / self.total_requests, 2) if self.total_requests > 0 else 0.0

            # Constrói ranking dos top 15 endpoints mais chamados
            top_routes = []
            for key, val in self.routes_stats.items():
                cnt = val["count"]
                avg = round(val["total_duration_ms"] / cnt, 2) if cnt > 0 else 0.0
                top_routes.append({
                    "route": key,
                    "method": val["method"],
                    "path": val["path"],
                    "count": cnt,
                    "avg_duration_ms": avg,
                    "errors": val["errors"],
                })

            top_routes.sort(key=lambda x: x["count"], reverse=True)

            return {
                "total_requests": self.total_requests,
                "requests_per_minute": rpm,
                "requests_per_second": rps,
                "avg_latency_ms": avg_latency,
                "status_counts": dict(self.status_counts),
                "top_endpoints": top_routes[:15],
                "recent_errors": list(reversed(list(self.recent_errors))),
                "uptime_seconds": int(now - self.start_time),
            }


class SpeedtestManager:
    def __init__(self):
        self.status: str = "idle"  # "idle" | "running" | "completed" | "error"
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.last_run_time: Optional[float] = None
        self._lock = asyncio.Lock()
        self._load_cache()

    def _load_cache(self):
        try:
            if os.path.exists(SPEEDTEST_CACHE_FILE):
                with open(SPEEDTEST_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.last_result = data.get("result")
                    self.last_run_time = data.get("timestamp")
                    if self.last_result:
                        self.status = "completed"
        except Exception as e:
            logger.warning(f"Erro ao carregar cache do speedtest: {e}")

    def _save_cache(self, result: Dict[str, Any]):
        try:
            with open(SPEEDTEST_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "result": result,
                    "timestamp": time.time(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache do speedtest: {e}")

    def _execute_sync_speedtest(self) -> Dict[str, Any]:
        import speedtest
        st = speedtest.Speedtest(secure=True)
        st.get_best_server()
        st.download(threads=2)
        st.upload(threads=2)
        res = st.results.dict()

        download_mbps = round(res.get("download", 0) / 1_000_000.0, 2)
        upload_mbps = round(res.get("upload", 0) / 1_000_000.0, 2)
        ping_ms = round(res.get("ping", 0), 2)
        server_info = res.get("server", {})
        client_info = res.get("client", {})

        return {
            "download_mbps": download_mbps,
            "upload_mbps": upload_mbps,
            "ping_ms": ping_ms,
            "server_name": server_info.get("name", "N/A"),
            "server_country": server_info.get("country", "N/A"),
            "server_sponsor": server_info.get("sponsor", "N/A"),
            "server_host": server_info.get("host", "N/A"),
            "isp": client_info.get("isp", "N/A"),
            "client_ip": client_info.get("ip", "N/A"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def run_test(self) -> Dict[str, Any]:
        if self.status == "running":
            return {"status": "running", "message": "Um teste de velocidade já está em execução no servidor."}

        async with self._lock:
            self.status = "running"
            self.last_error = None

        try:
            # Executa em thread separada para não travar o loop assíncrono do FastAPI
            result = await asyncio.to_thread(self._execute_sync_speedtest)
            self.last_result = result
            self.last_run_time = time.time()
            self.status = "completed"
            self._save_cache(result)
            return {"status": "completed", "result": result}
        except Exception as e:
            logger.error(f"Falha ao executar teste de velocidade de internet: {e}")
            self.status = "error"
            self.last_error = str(e)
            return {"status": "error", "error": str(e)}

    def get_state(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "last_run_timestamp": self.last_run_time,
        }


def format_uptime(seconds: float) -> str:
    secs = int(seconds)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "menos de 1m"


def get_server_system_metrics() -> Dict[str, Any]:
    # Métricas de CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count_logical = psutil.cpu_count(logical=True) or 1
    cpu_count_physical = psutil.cpu_count(logical=False) or cpu_count_logical

    # Métricas de Memória RAM
    mem = psutil.virtual_memory()
    total_mem_mb = round(mem.total / (1024 * 1024), 2)
    used_mem_mb = round(mem.used / (1024 * 1024), 2)
    available_mem_mb = round(mem.available / (1024 * 1024), 2)

    # Métricas de Disco (Root)
    try:
        root_path = os.path.abspath(os.sep)
        disk = psutil.disk_usage(root_path)
        disk_total_gb = round(disk.total / (1024 ** 3), 2)
        disk_used_gb = round(disk.used / (1024 ** 3), 2)
        disk_free_gb = round(disk.free / (1024 ** 3), 2)
        disk_percent = disk.percent
    except Exception:
        disk_total_gb = 0.0
        disk_used_gb = 0.0
        disk_free_gb = 0.0
        disk_percent = 0.0

    # Rede (Estatísticas de IO)
    net_io = psutil.net_io_counters()
    net_sent_mb = round(net_io.bytes_sent / (1024 * 1024), 2)
    net_recv_mb = round(net_io.bytes_recv / (1024 * 1024), 2)

    # Processo Atual do OmniWatch
    try:
        current_proc = psutil.Process()
        proc_mem_mb = round(current_proc.memory_info().rss / (1024 * 1024), 2)
        proc_cpu_percent = current_proc.cpu_percent(interval=None)
        proc_threads = current_proc.num_threads()
    except Exception:
        proc_mem_mb = 0.0
        proc_cpu_percent = 0.0
        proc_threads = 0

    # Uptime do SO
    boot_time = psutil.boot_time()
    so_uptime_seconds = time.time() - boot_time

    return {
        "cpu": {
            "percent": cpu_percent,
            "logical_cores": cpu_count_logical,
            "physical_cores": cpu_count_physical,
        },
        "memory": {
            "total_mb": total_mem_mb,
            "used_mb": used_mem_mb,
            "available_mb": available_mem_mb,
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": disk_total_gb,
            "used_gb": disk_used_gb,
            "free_gb": disk_free_gb,
            "percent": disk_percent,
        },
        "network": {
            "sent_mb": net_sent_mb,
            "recv_mb": net_recv_mb,
        },
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "uptime_seconds": int(so_uptime_seconds),
            "uptime_formatted": format_uptime(so_uptime_seconds),
            "process_count": len(psutil.pids()),
        },
        "app_process": {
            "memory_mb": proc_mem_mb,
            "cpu_percent": proc_cpu_percent,
            "threads": proc_threads,
        }
    }


# Instâncias globais únicas para rastreamento
api_metrics_tracker = APIMetricsTracker()
speedtest_manager = SpeedtestManager()
