import importlib.util
import subprocess
import sys

# 在子程序檢查，避免安裝前先載入目前 kernel 的動態函式庫
check_code = """
import pygmt, pandas
import shutil
from pygmt.clib import Session
assert pygmt.__version__.lstrip('v').startswith('0.17.')
assert shutil.which('gs')
with Session() as session:
    assert session.info['version'].startswith('6.5.')
"""
try:
    ENV_READY = subprocess.run(
        [sys.executable, "-c", check_code], capture_output=True, timeout=30
    ).returncode == 0
except subprocess.TimeoutExpired:
    ENV_READY = False

IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
if ENV_READY:
    print("環境已可用，跳過安裝。")
elif IN_COLAB:
    print("步驟 2/2：安裝 PyGMT 與相依套件（約 2–4 分鐘），下方會逐行顯示進度。", flush=True)
    command = ["mamba", "install", "-y", "-c", "conda-forge", "pygmt=0.17", "gmt=6.5", "ghostscript=10.04", "pandas"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"安裝失敗（exit {process.returncode}），請重新執行本格或重啟執行環境。")
    print("安裝完成，可以往下執行。")
else:
    print("跳過 Colab 安裝。")
