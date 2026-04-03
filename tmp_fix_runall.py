# -*- coding: utf-8 -*-
import json
from pathlib import Path

p = Path(r"D:\Code\siaug-2025\siaug\notebooks\nih-binary-error-analysis.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
nb["cells"][17]["source"] = [
    "RUN_ALL = False\n",
    "\n",
    "all_results = {}\n",
    "if RUN_ALL:\n",
    "    for key in EXPERIMENTS:\n",
    '        print(f"\\n=== {key} ===")\n',
    "        all_results[key] = collect_predictions(\n",
    "            experiment_key=key,\n",
    "            threshold=THRESHOLD,\n",
    "            batch_size=BATCH_SIZE,\n",
    "            num_workers=NUM_WORKERS,\n",
    "            device=DEVICE,\n",
    "        )\n",
    "        display(summarize_errors(all_results[key]))\n",
]
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("RUN_ALL fixed")
