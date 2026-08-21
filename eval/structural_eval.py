import argparse,json,statistics
from pathlib import Path
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]
    report={"examples":len(rows),"empty_rate":sum(not r.get("answer","").strip() for r in rows)/max(1,len(rows)),"mean_latency":statistics.mean([r.get("latency_seconds",0) for r in rows]) if rows else 0,"retrieval_status":{k:sum(r.get("retrieval_status")==k for r in rows) for k in ["sufficient","partial","no_evidence","disabled"]}}
    print(json.dumps(report,indent=2))

