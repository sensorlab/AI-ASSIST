import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Final

from dotenv import load_dotenv

load_dotenv()

import httpx
import pandas as pd
from tqdm.auto import tqdm

from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
API_ENDPOINT: Final[str] = "http://localhost:8000/api/v1/estimate/tsa/by-generator"


@dataclass(frozen=True)
class QueryResult:
    state_id: Any
    elapsed_s: float
    status_code: int | None
    ok: bool
    error: str | None = None


def _configured_lf_path() -> Path:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir

    lf_path, _, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path


def _query_state(
    state_id: Any,
    state: pd.Series,
    *,
    endpoint: str,
    variant: str,
    exclude_self: bool,
    timeout_s: float | None,
    max_states: int | None = None,
) -> QueryResult:
    payload: dict[str, Any] = {
        "variant": variant,
        "state": state.to_dict(),
    }
    if exclude_self:
        payload["exclude_uids"] = [str(state_id)]
    if max_states is not None:
        payload["max_states"] = max_states

    started = time.perf_counter()
    status_code: int | None = None
    try:
        with httpx.Client(timeout=timeout_s, http2=True) as client:
            response = client.post(endpoint, json=payload)
            status_code = response.status_code
            response.raise_for_status()
    except Exception as exc:
        return QueryResult(
            state_id=state_id,
            elapsed_s=time.perf_counter() - started,
            status_code=status_code,
            ok=False,
            error=str(exc),
        )

    return QueryResult(
        state_id=state_id,
        elapsed_s=time.perf_counter() - started,
        status_code=status_code,
        ok=True,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")

    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_latency_benchmark(
    lf: pd.DataFrame,
    *,
    n_samples: int,
    concurrency: int,
    endpoint: str,
    variant: str,
    exclude_self: bool,
    timeout_s: float | None,
    max_states: int | None = None,
) -> list[QueryResult]:
    if n_samples <= 0:
        raise ValueError("`n_samples` must be greater than 0")
    if concurrency <= 0:
        raise ValueError("`concurrency` must be greater than 0")

    samples = list(lf.head(n_samples).iterrows())
    if not samples:
        raise ValueError("No LF samples available to query")

    results: list[QueryResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _query_state,
                state_id,
                state,
                endpoint=endpoint,
                variant=variant,
                exclude_self=exclude_self,
                timeout_s=timeout_s,
                max_states=max_states,
            )
            for state_id, state in samples
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying states"):
            results.append(future.result())

    return results


def print_summary(results: list[QueryResult], *, wall_time_s: float) -> None:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.elapsed_s for result in successes]

    print("\nLatency benchmark summary")
    print("=========================")
    print(f"requests_total={len(results)}")
    print(f"requests_success={len(successes)}")
    print(f"requests_failed={len(failures)}")
    print(f"wall_time_s={wall_time_s:.6f}")
    print(f"throughput_req_s={len(results) / wall_time_s:.3f}")

    if latencies:
        print(f"latency_min_s={min(latencies):.6f}")
        print(f"latency_mean_s={mean(latencies):.6f}")
        print(f"latency_median_s={median(latencies):.6f}")
        print(f"latency_p95_s={_percentile(latencies, 0.95):.6f}")
        print(f"latency_p99_s={_percentile(latencies, 0.99):.6f}")
        print(f"latency_max_s={max(latencies):.6f}")

    if failures:
        print("\nFailures")
        print("========")
        for result in failures[:10]:
            print(f"state={result.state_id!r} status={result.status_code} error={result.error}")
        if len(failures) > 10:
            print(f"... {len(failures) - 10} more failures omitted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure service query latency for the first N LF samples.")
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--endpoint", default=API_ENDPOINT)
    parser.add_argument("--variant", default="1.0.0")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help=(
            "StateRequest.max_states for each request. Left unset the service uses its own "
            "default. Worth sweeping on the SSSA route, whose cross-state mode matching is "
            "roughly quadratic in retrieved rows - the reason max_states is capped at 500."
        ),
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Do not exclude the queried state from retrieval. By default the state excludes itself.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    lf_path = _configured_lf_path()

    logger.info(f"Latency benchmark dataset: lf={lf_path}")
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Samples: first {args.n_samples}; concurrency={args.concurrency}")

    lf = pd.read_pickle(lf_path)

    started = time.perf_counter()
    results = run_latency_benchmark(
        lf,
        n_samples=args.n_samples,
        concurrency=args.concurrency,
        endpoint=args.endpoint,
        variant=args.variant,
        exclude_self=not args.include_self,
        timeout_s=args.timeout_s,
        max_states=args.max_states,
    )
    print_summary(results, wall_time_s=time.perf_counter() - started)


if __name__ == "__main__":
    main()
