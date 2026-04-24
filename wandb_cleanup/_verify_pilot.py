"""Spot-check that the 4 pilot runs actually have axes/<ID>_<Name> values."""

import wandb

api = wandb.Api()
IDS = ["1iuxis2l", "7ipzizh6", "wqo0y6mw", "17ovy6fy"]

for rid in IDS:
    run = api.run(f"nterlind-nikhef/thesis-ml/{rid}")
    cfg = run.config
    if isinstance(cfg, str):
        import json

        cfg = json.loads(cfg)
    new_v2 = {k: v for k, v in dict(cfg).items() if k.startswith("axes/") and "_" in k[5:] and k[5:].split("_", 1)[0][0] in "ABCDEFGHLPRTZ"}
    # Filter to keys of the form axes/<UPPER+digit+dash>_<Name with space>
    new_v2 = {k: v for k, v in new_v2.items() if any(ch.isupper() for ch in k[5:].split("_", 1)[0]) and any(ch.isdigit() for ch in k[5:].split("_", 1)[0])}
    non_empty = {k: v for k, v in new_v2.items() if v not in (None, "", {})}
    print(f"\n=== {rid}  ({run.name}) ===")
    print(f"  total V2 columns present : {len(new_v2)}")
    print(f"  V2 columns with a value  : {len(non_empty)}")
    print("  sample values:")
    for k in sorted(non_empty)[:10]:
        print(f"    {k:60s} = {non_empty[k]!r}")
    empty_sample = [k for k in sorted(new_v2) if k not in non_empty][:5]
    if empty_sample:
        print("  sample empty columns:")
        for k in empty_sample:
            print(f"    {k}")
