import json
from pathlib import Path

data_path = Path("./data/ShapeNetCore.v2")

counts = dict()
for path in data_path.iterdir():
    if path.is_dir():
        counts[path.name] = len(list(path.iterdir()))


top_10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

sum_count = 0
for name, count in top_10:
    sum_count += count
    taxonomy_file = data_path / "taxonomy.json"
    with open(taxonomy_file, "r") as f:
        taxonomy = json.load(f)

    print({t["name"] for t in taxonomy})
    break


print("total:", sum_count)
