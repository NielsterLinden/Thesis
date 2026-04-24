"""Read-only probe: can the public API still fetch the 4 pilot runs?"""

import json

import wandb

IDS = ["1iuxis2l", "7ipzizh6", "wqo0y6mw", "17ovy6fy"]
api = wandb.Api()

for rid in IDS:
    print(f"\n=== {rid} ===")
    try:
        run = api.run(f"nterlind-nikhef/thesis-ml/{rid}")
        print(f"  name           : {run.name}")
        print(f"  state          : {run.state}")
        cfg = run._attrs.get("config")
        print(f"  _attrs[config] type : {type(cfg).__name__}")
        if isinstance(cfg, str):
            print(f"  _attrs[config] len  : {len(cfg)} chars")
            try:
                parsed = json.loads(cfg)
                print(f"  parsed keys         : {len(parsed)}")
                v2 = [k for k in parsed if isinstance(k, str) and k.startswith("axes/") and any(ch.isdigit() for ch in k[5:].split("_", 1)[0])]
                print(f"  V2 keys present     : {len(v2)}")
                print(f"  sample V2           : {v2[:5]}")
            except Exception as e:
                print(f"  json.loads failed   : {e!r}")
        elif isinstance(cfg, dict):
            print(f"  dict keys           : {len(cfg)}")
            v2 = [k for k in cfg if isinstance(k, str) and k.startswith("axes/") and any(ch.isdigit() for ch in k[5:].split("_", 1)[0])]
            print(f"  V2 keys present     : {len(v2)}")
            print(f"  sample V2           : {v2[:5]}")
    except Exception as e:
        print(f"  FETCH FAILED: {type(e).__name__}: {e}")
