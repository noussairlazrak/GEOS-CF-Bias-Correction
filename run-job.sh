# run-job.sh
# ----
#!/usr/bin/env bash

# Debugging container image
if [ -n "${ECS_CONTAINER_METADATA_URI_V4-}" ]; then
  uv run python - <<'PY'
import json
import os
import urllib.request

url = os.environ["ECS_CONTAINER_METADATA_URI_V4"]
with urllib.request.urlopen(url) as resp:
    data = json.load(resp)

print(f"container_image={data.get('Image', '')}")
print(f"container_image_id={data.get('ImageID', '')}")
print(f"container_name={data.get('Name', '')}")
print(f"docker_id={data.get('DockerId', '')}")

# import rasterio

PY
fi

# Run the actual commands
uv run python -u generate.py --skip-plotting --skip-openaq --s3-only --model-cache s3
uv run python -u compress.py
