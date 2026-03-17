"""Self-terminate the current RunPod pod."""

import os
import runpod

runpod.api_key = os.environ.get("RUNPOD_API_KEY")
pod_id = os.environ.get("RUNPOD_POD_ID")

if runpod.api_key and pod_id:
    print(f"=== pod {pod_id} self-terminating ===")
    runpod.terminate_pod(pod_id)
else:
    print("RUNPOD_API_KEY or RUNPOD_POD_ID not set, skipping auto-terminate")
