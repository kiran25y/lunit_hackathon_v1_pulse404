import argparse, asyncio, json, uuid
from pathlib import Path
from .main import harness

async def run(input_path,output_path):
    h=harness(); out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    with Path(input_path).open(encoding="utf-8") as src,out.open("w",encoding="utf-8") as dst:
        for line in src:
            row=json.loads(line); rid=str(row.get("id",uuid.uuid4())); result=await h.answer(row["messages"]); h.save_trace(rid,result)
            dst.write(json.dumps({"id":rid,"answer":result.answer,"retrieval_status":result.retrieval_status,"latency_seconds":result.trace["latency_seconds"]},ensure_ascii=False)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args(); asyncio.run(run(a.input,a.output))

