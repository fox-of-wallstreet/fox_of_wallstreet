import subprocess
import os
import sys
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

app = FastAPI(title="Fox of Wallstreet Task Runner")

import json
from fastapi import HTTPException
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('.')

from core.tools import fnline

app = FastAPI()

ARTIFACTS_DIR = Path("artifacts")

@app.get("/available-models")
async def get_available_models():
    """Returns a list of folder names that have all 3 required files."""
    valid_models = []
    required = {"model.zip", "scaler.pkl", "metadata.json"}
    
    if not ARTIFACTS_DIR.exists():
        return {"models": []}

    for folder in ARTIFACTS_DIR.iterdir():
        if folder.is_dir():
            files = {f.name for f in folder.iterdir() if f.is_file()}
            if required.issubset(files):
                valid_models.append(folder.name)
    return {"models": valid_models}

@app.get("/model-details/{model_name}")
async def get_model_details(model_name: str):
    """Reads and returns the content of metadata.json for a specific model."""
    metadata_path = ARTIFACTS_DIR / model_name / "metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found for this model")
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")

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

def setup_artifact_symlinks():
    """
    Synchronizes the 'artifacts' directory with directories found in 'preloaded'.
    Expects both to be at the same level in the project root.
    """
    source_dir = Path("preloaded")
    target_dir = Path("artifacts")

    # 1. Ensure directories exist
    if not source_dir.exists():
        print(fnline(), f"[!] Warning: Source directory {source_dir} not found.")
        return

    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        print(fnline(), f"[*] Created target directory: {target_dir}")

    # 2. Iterate through 'preloaded'
    for item in source_dir.iterdir():
        if item.is_dir():
            link_name = target_dir / item.name
            
            # Use relative pathing for the symlink (more portable)
            # This points from 'artifacts/dir' back to '../preloaded/dir'
            relative_source = os.path.join("..", "preloaded", item.name)

            if not link_name.exists():
                try:
                    os.symlink(relative_source, link_name)
                    print(fnline(), f"[✓] Linked: artifacts/{item.name} -> {relative_source}")
                except OSError as e:
                    print(fnline(), f"[X] Failed to link {item.name}: {e}")
            else:
                print(fnline(), f"[-] {item.name} already exists in artifacts, skipping.")

if __name__ == "__main__":
    import uvicorn
    setup_artifact_symlinks()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
