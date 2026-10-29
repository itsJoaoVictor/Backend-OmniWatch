from typing import List, Optional, Literal
from pydantic import BaseModel


class ComponentHealth(BaseModel):
    id: str
    name: str
    description: str
    category: str
    status: Literal["operational", "degraded", "outage"]
    latency_ms: float
    message: str


class UptimeDay(BaseModel):
    date: str
    status: Literal["operational", "degraded", "outage"]
    uptime_pct: float


class HealthSummaryResponse(BaseModel):
    overall_status: Literal["operational", "degraded", "outage"]
    version: str
    timestamp: str
    uptime_seconds: int
    uptime_formatted: str
    uptime_percentage_90d: float
    avg_latency_ms: float
    components: List[ComponentHealth]
    history_90d: List[UptimeDay]
