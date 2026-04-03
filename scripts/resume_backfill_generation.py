"""Retry a backfill generation run until it completes successfully."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Sequence


def build_backfill_command(
    *,
    python_bin: Path,
    backfill_script: Path,
    model_dir: Path,
    model_name: str,
    tasks_root: Path,
    client_type: str,
    api_key_env: str,
    api_base_url: str | None,
    anthropic_version: str,
    target_trials: int,
    temperature: float,
    max_output_tokens: int,
    request_timeout_seconds: float,
    concurrency: int,
    supplement_tag: str,
) -> list[str]:
    command = [
        str(python_bin),
        str(backfill_script),
        "--model-dir",
        str(model_dir),
        "--model-name",
        model_name,
        "--tasks-root",
        str(tasks_root),
        "--client",
        client_type,
        "--api-key-env",
        api_key_env,
        "--target-trials",
        str(target_trials),
        "--temperature",
        str(temperature),
        "--max-output-tokens",
        str(max_output_tokens),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
        "--concurrency",
        str(concurrency),
        "--anthropic-version",
        anthropic_version,
        "--supplement-tag",
        supplement_tag,
    ]
    if api_base_url:
        command.extend(["--api-base-url", api_base_url])
    return command


def resume_backfill_until_complete(
    command: Sequence[str],
    *,
    cwd: Path,
    retry_delay_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[int], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    attempt = 1
    while True:
        log(f"[resume] attempt {attempt} starting")
        result = runner(
            list(command),
            cwd=cwd,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            log(f"[resume] attempt {attempt} completed successfully")
            return 0
        log(
            f"[resume] attempt {attempt} exited code={result.returncode}; retrying in {retry_delay_seconds}s"
        )
        attempt += 1
        sleeper(retry_delay_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume backfill generation after failures.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Curated model directory")
    parser.add_argument("--model-name", required=True, help="Provider model name")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmark_data/tasks"),
        help="Root directory containing benchmark tasks",
    )
    parser.add_argument(
        "--client",
        choices=("openai-compatible", "anthropic-native", "gemini-native"),
        default="openai-compatible",
        help="Generation client backend used for backfill runs",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Optional base URL for the selected client",
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Anthropic API version when using --client anthropic-native",
    )
    parser.add_argument("--target-trials", type=int, default=20, help="Desired trials per task")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens for backfill generation; use 0 to omit the limit",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP request timeout used by the generation client",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent generation requests")
    parser.add_argument(
        "--supplement-tag",
        default="max_tokens_2048",
        help="Compatibility label passed through to the backfill script",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=int,
        default=60,
        help="Delay before restarting after a failed backfill attempt",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    command = build_backfill_command(
        python_bin=Path(sys.executable),
        backfill_script=project_root / "scripts" / "backfill_generation_records.py",
        model_dir=args.model_dir.resolve(),
        model_name=args.model_name,
        tasks_root=args.tasks_root.resolve(),
        client_type=args.client,
        api_key_env=args.api_key_env,
        api_base_url=args.api_base_url,
        anthropic_version=args.anthropic_version,
        target_trials=args.target_trials,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        concurrency=args.concurrency,
        supplement_tag=args.supplement_tag,
    )
    try:
        return resume_backfill_until_complete(
            command,
            cwd=project_root,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except KeyboardInterrupt:
        print("[resume] interrupted by user", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
