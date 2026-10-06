"""
Coût de calcul mesuré : latence, temps CPU, mémoire, énergie.

Énergie : compteur RAPL du processeur (/sys/class/powercap) quand il est lisible ; sinon
estimation explicite  E = temps CPU × puissance par cœur occupé (hypothèse affichée).
VRAM : aucun calcul GPU (NumPy sur CPU) ; FLOPs d'un LLM distant : non mesurables côté client.
"""
from __future__ import annotations

import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

from aiotech.bench.metrics import percentile

RAPL_FILE = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
DEFAULT_CPU_WATTS = 15.0


def read_rapl_uj() -> int | None:
    try:
        return int(RAPL_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024, 1)


def host_info() -> dict[str, str | int]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "processor": platform.processor() or platform.machine(),
    }


@dataclass
class Meter:
    """Accumule latences et temps CPU d'un système ; énergie RAPL si disponible."""

    latencies_ms: list[float] = field(default_factory=list)
    cpu_seconds: float = 0.0
    rapl_uj: int = 0
    rapl_ok: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    llm_calls: int = 0
    llm_latencies_ms: list[float] = field(default_factory=list)

    def start(self) -> tuple[float, float, int | None]:
        return time.perf_counter(), time.process_time(), read_rapl_uj()

    def stop(self, started: tuple[float, float, int | None]) -> None:
        wall, cpu, rapl = started
        self.latencies_ms.append((time.perf_counter() - wall) * 1000.0)
        self.cpu_seconds += time.process_time() - cpu
        after = read_rapl_uj()
        if rapl is None or after is None or after < rapl:
            self.rapl_ok = False
        else:
            self.rapl_uj += after - rapl

    def add_llm(self, tokens_in: int, tokens_out: int, cost: float | None, latency_ms: float) -> None:
        self.llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.llm_latencies_ms.append(latency_ms)
        if cost is None:
            self.cost_known = False
        else:
            self.cost_usd += cost

    def summary(self, cpu_watts: float, python_peak_mb: float | None) -> dict[str, object]:
        n = max(1, len(self.latencies_ms))
        if self.rapl_ok and self.rapl_uj > 0:
            energy_mwh = self.rapl_uj / 1e6 / 3600.0 * 1000.0 / n
            energy = {"value_mwh": round(energy_mwh, 5), "method": "RAPL (paquet processeur, mesuré)"}
        else:
            energy_mwh = self.cpu_seconds * cpu_watts / 3600.0 * 1000.0 / n
            energy = {"value_mwh": round(energy_mwh, 5),
                      "method": f"estimation : temps CPU × {cpu_watts:g} W par cœur occupé"}
        return {
            "queries": len(self.latencies_ms),
            "latency_ms": {"p50": round(percentile(self.latencies_ms, 50), 2),
                           "p95": round(percentile(self.latencies_ms, 95), 2),
                           "mean": round(sum(self.latencies_ms) / n, 2)},
            "cpu_ms_per_query": round(self.cpu_seconds * 1000.0 / n, 2),
            "energy_per_query": energy,
            "python_peak_mb": python_peak_mb,
            "vram_mb": 0,
            "llm_calls": self.llm_calls,
            "llm_latency_ms": {"p50": round(percentile(self.llm_latencies_ms, 50), 1),
                               "p95": round(percentile(self.llm_latencies_ms, 95), 1)} if self.llm_latencies_ms else None,
            "tokens_in_per_query": round(self.tokens_in / n, 1),
            "tokens_out_per_query": round(self.tokens_out / n, 1),
            "cost_usd_per_query": round(self.cost_usd / n, 6) if self.cost_known else None,
        }


class PythonPeak:
    """Pic d'allocation Python (tracemalloc) d'un bloc ; mesuré à part car tracemalloc ralentit le code."""

    def __enter__(self) -> PythonPeak:
        tracemalloc.start()
        self.peak_mb = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = round(peak / (1024 * 1024), 2)
