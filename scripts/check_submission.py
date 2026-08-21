from pathlib import Path
import re,sys
bad=[]
for p in Path(".").rglob("*"):
    if p.is_file() and not any(x in p.parts for x in (".git",".venv")) and p.stat().st_size<2_000_000:
        text=p.read_text(errors="ignore")
        if re.search(r"sk-[A-Za-z0-9_-]{20,}|Bearer\s+(?!REPLACE)[A-Za-z0-9._-]{20,}",text): bad.append(str(p))
required=["README.md","app/pipeline.py","app/l2_client.py","app/mcp_client.py","config/mcp_tools.example.json"]; missing=[x for x in required if not Path(x).exists()]
if bad or missing: print({"possible_secrets":bad,"missing":missing}); sys.exit(1)
print("Submission structure and basic secret scan passed.")

