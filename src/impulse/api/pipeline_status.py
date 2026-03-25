"""GET /jobs/{jobId}/pipeline-status -- Fetch Step Functions execution status and history."""

from __future__ import annotations

import json

import boto3
from loguru import logger

from impulse.db.client import get_collection


def get_pipeline_status(job_id: str, user_id: str) -> dict:
    """Return the Step Functions execution status, step history, and error details."""
    jobs_collection = get_collection("jobs")
    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0, "step_functions_arn": 1, "status": 1},
    )

    if not job:
        return _response(404, {"error": "Job not found"})

    execution_arn = job.get("step_functions_arn", "")
    if not execution_arn:
        return _response(
            200,
            {
                "execution_status": "NOT_STARTED",
                "steps": [],
                "error": None,
                "map_run": None,
            },
        )

    sfn = boto3.client("stepfunctions")

    # 1. Get execution overview
    try:
        execution = sfn.describe_execution(executionArn=execution_arn)
    except Exception as e:
        logger.warning(f"Cannot describe execution {execution_arn}: {e}")
        return _response(
            200,
            {
                "execution_status": "UNKNOWN",
                "steps": [],
                "error": str(e),
                "map_run": None,
            },
        )

    exec_status = execution.get("status", "UNKNOWN")
    exec_error = execution.get("error")
    exec_cause = execution.get("cause")

    # 2. Get step-by-step history
    steps: list[dict] = []
    try:
        history = sfn.get_execution_history(
            executionArn=execution_arn,
            maxResults=200,
            reverseOrder=False,
        )
        steps = _parse_history(history.get("events", []))
    except Exception as e:
        logger.warning(f"Cannot get history for {execution_arn}: {e}")

    # 3. If there's a DistributedMap, get its run summary
    map_run_info = None
    try:
        map_runs = sfn.list_map_runs(executionArn=execution_arn)
        for run in map_runs.get("mapRuns", []):
            run_detail = sfn.describe_map_run(mapRunArn=run["mapRunArn"])
            item_counts = run_detail.get("itemCounts", {})
            exec_counts = run_detail.get("executionCounts", {})
            map_run_info = {
                "status": run_detail.get("status", "UNKNOWN"),
                "total": item_counts.get("total", 0),
                "succeeded": item_counts.get("succeeded", 0),
                "failed": item_counts.get("failed", 0),
                "pending": item_counts.get("pending", 0),
                "running": item_counts.get("running", 0),
                "aborted": item_counts.get("aborted", 0),
            }

            # Get failed child execution errors
            if item_counts.get("failed", 0) > 0:
                failed_children = _get_failed_children(sfn, run["mapRunArn"])
                map_run_info["failed_items"] = failed_children
    except Exception as e:
        logger.warning(f"Cannot get map runs: {e}")

    result = {
        "execution_status": exec_status,
        "started_at": execution.get("startDate", ""),
        "stopped_at": execution.get("stopDate", ""),
        "steps": steps,
        "error": exec_error,
        "cause": exec_cause,
        "map_run": map_run_info,
    }

    return _response(200, result)


def _parse_history(events: list[dict]) -> list[dict]:
    """Convert Step Functions history events into a simplified step list."""
    steps: list[dict] = []
    step_map: dict[str, dict] = {}  # track steps by id

    for event in events:
        etype = event.get("type", "")
        eid = event.get("id", 0)
        ts = event.get("timestamp", "")

        # Step entered
        if "StateEntered" in etype:
            details = event.get("stateEnteredEventDetails", {})
            name = details.get("name", "Unknown")
            step = {
                "name": name,
                "status": "RUNNING",
                "started_at": ts,
                "ended_at": None,
                "input_preview": _truncate(details.get("input", ""), 300),
                "output_preview": None,
                "error": None,
                "cause": None,
            }
            step_map[name] = step
            steps.append(step)

        # Step exited successfully
        elif "StateExited" in etype:
            details = event.get("stateExitedEventDetails", {})
            name = details.get("name", "")
            if name in step_map:
                step_map[name]["status"] = "SUCCEEDED"
                step_map[name]["ended_at"] = ts
                step_map[name]["output_preview"] = _truncate(
                    details.get("output", ""), 300
                )

        # Task failed
        elif etype == "TaskFailed":
            details = event.get("taskFailedEventDetails", {})
            # Find the most recent running step
            for s in reversed(steps):
                if s["status"] == "RUNNING":
                    s["status"] = "FAILED"
                    s["error"] = details.get("error", "")
                    s["cause"] = _truncate(details.get("cause", ""), 1000)
                    s["ended_at"] = ts
                    break

        # Lambda function failed
        elif etype == "LambdaFunctionFailed":
            details = event.get("lambdaFunctionFailedEventDetails", {})
            for s in reversed(steps):
                if s["status"] == "RUNNING":
                    s["status"] = "FAILED"
                    s["error"] = details.get("error", "")
                    s["cause"] = _truncate(details.get("cause", ""), 1000)
                    s["ended_at"] = ts
                    break

        # MapRun failed
        elif etype == "MapRunFailed":
            details = event.get("mapRunFailedEventDetails", {})
            for s in reversed(steps):
                if s["status"] == "RUNNING":
                    s["status"] = "FAILED"
                    s["error"] = details.get("error", "")
                    s["cause"] = _truncate(details.get("cause", ""), 1000)
                    s["ended_at"] = ts
                    break

        # Execution failed
        elif etype == "ExecutionFailed":
            details = event.get("executionFailedEventDetails", {})
            # Mark any still-running steps as failed
            for s in steps:
                if s["status"] == "RUNNING":
                    s["status"] = "FAILED"
                    s["error"] = details.get("error", "")
                    s["cause"] = _truncate(details.get("cause", ""), 1000)
                    s["ended_at"] = ts

    return steps


def _get_failed_children(sfn_client, map_run_arn: str) -> list[dict]:
    """Get error details from failed child executions of a DistributedMap."""
    failed_items: list[dict] = []

    try:
        result = sfn_client.list_executions(
            mapRunArn=map_run_arn,
            statusFilter="FAILED",
            maxResults=10,
        )
        for child in result.get("executions", []):
            child_arn = child["executionArn"]
            try:
                child_detail = sfn_client.describe_execution(executionArn=child_arn)
                child_input = child_detail.get("input", "{}")
                try:
                    parsed_input = json.loads(child_input)
                except json.JSONDecodeError:
                    parsed_input = {}

                failed_items.append(
                    {
                        "document_key": parsed_input.get("document_key", "unknown"),
                        "task_type": parsed_input.get("task_type", "unknown"),
                        "error": child_detail.get("error", ""),
                        "cause": _truncate(child_detail.get("cause", ""), 500),
                    }
                )
            except Exception:
                failed_items.append(
                    {
                        "document_key": "unknown",
                        "error": "Could not fetch details",
                        "cause": "",
                    }
                )
    except Exception as e:
        logger.warning(f"Cannot list failed children: {e}")

    return failed_items


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string and add ellipsis if needed."""
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }
