import os, sys, subprocess, time
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository
AGENT_DIR = _Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else WORK.parent / "agent"      # an agent folder with the model stages
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(AGENT_DIR))
os.environ.update({"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1", "AO_LLM_PREFIX": "OPENAI", "OPENAI_API_KEY": "test",
                   "OPENAI_BASE_URL": "http://127.0.0.1:8767/v1", "OPENAI_MODEL": "mock"})      # the prefix keeps real keys in the environment out of it
from pathlib import Path
from challenge.v4_workflow import V4Workflow
import agent as A
init = V4Workflow(WORK / "cards/dev/d101").initialize_payload(900.0, 1.0)
for mode, expect in (("good", True), ("veto", False), ("garbage", True), ("error", True)):
    proc = subprocess.Popen([sys.executable, str(WORK / "tools/mock_llm.py"), "--port", "8767", "--mode", mode]); time.sleep(0.8)
    ag = A.Agent(init)
    payload = {"wallclock": {"wall_remaining_seconds": 1700.0}}
    now = ag.survey.nights[5][0] + 3600
    t = time.time()
    got = ag.supervisor.allow_report(payload, now, 100.0, 0.7, 0)
    print(f"{mode:8s} allow_report -> {got} (expected {expect}) in {time.time() - t:.1f}s; veto_until {ag.supervisor.veto_until_hours}")
    assert got == expect
    proc.kill(); time.sleep(0.3)
print("verdict path ok")
