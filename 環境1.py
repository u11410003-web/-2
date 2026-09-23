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
    print("步驟 1/2：安裝 Conda（約 1 分鐘）。完成後 Colab 會自動重啟執行環境，等重新連線再執行下一格。", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "condacolab==0.1.13"])
    import condacolab
    condacolab.install()
else:
    print("本機模式：使用目前 Python 環境。")
