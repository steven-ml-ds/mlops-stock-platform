"""Weekly retrain pipeline: fetch fresh data, train a challenger, let it
fight the champion for the @production alias.

Each task is a thin wrapper around platform_core so the logic stays
testable outside Airflow; the DAG only handles orchestration and XCom.
"""

from datetime import datetime

from airflow.sdk import dag, task

DEFAULT_MIN_EDGE = 0.0


@dag(
    dag_id="weekly_retrain",
    schedule="@weekly",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["mlops", "retrain"],
)
def weekly_retrain():
    @task
    def fetch_data() -> int:
        from platform_core.data import fetch_all

        frames = fetch_all(refresh=True)
        return sum(len(f) for f in frames.values())

    @task
    def train_challenger(n_rows: int) -> str:
        from platform_core.promote import DEFAULT_PARAMS
        from platform_core.train import train_once

        return train_once(DEFAULT_PARAMS, run_name="weekly-challenger", register=True)

    @task
    def promote_if_better(challenger_run_id: str) -> dict:
        from platform_core.promote import compare_and_promote

        return compare_and_promote(challenger_run_id, min_edge=DEFAULT_MIN_EDGE)

    promote_if_better(train_challenger(fetch_data()))


weekly_retrain()
