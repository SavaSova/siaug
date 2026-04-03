# -*- coding: utf-8 -*-
import json
from pathlib import Path

p = Path(r"D:\Code\siaug-2025\siaug\notebooks\nih-binary-error-analysis.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
text = "".join(nb["cells"][3]["source"])
old = """        ax.set_title(
            f"{row['file_name']}
"
            f"истинная метка={row['target']}, предсказание={row['prediction']}
"
            f"вероятность={row['probability']:.4f}",
            fontsize=10,
        )
"""
new = """        ax.set_title(
            f"{row['file_name']}\\n"
            f"истинная метка={row['target']}, предсказание={row['prediction']}\\n"
            f"вероятность={row['probability']:.4f}",
            fontsize=10,
        )
"""
if old not in text:
    raise SystemExit("broken f-string pattern not found")
text = text.replace(old, new)
nb["cells"][3]["source"] = [line + "\n" for line in text.split("\n") if line != ""]
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("show_examples title fixed")
