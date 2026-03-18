import subprocess
import os
import sys
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

app = FastAPI(title="Fox of Wallstreet Task Runner")

def stream_process(script_name: str, args: str):
    """Core logic to run a script and pipe output line-by-line."""
    script_path = os.path.join("scripts", script_name)
    
    def generate():
        if not os.path.exists(script_path):
            yield f"❌ Error: Script {script_path} not found.\n"
            return

        # -u ensures Python doesn't buffer logs
        command = [sys.executable, "-u", script_path]
        if args:
            command.extend(args.split())

        yield f"🚀 Executing: {' '.join(command)}\n"
        yield "------------------------------------------\n"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            yield line

        process.stdout.close()
        exit_code = process.wait()
        yield "------------------------------------------\n"
        yield f"✅ Finished with Exit Code: {exit_code}\n"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/data")
async def run_data(params: str = Query("")):
    return stream_process("data_engine.py", params)

@app.get("/train")
async def run_train(params: str = Query("")):
    return stream_process("train.py", params)

@app.get("/trade")
async def run_trade(params: str = Query("")):
    return stream_process("live_trader.py", params)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
