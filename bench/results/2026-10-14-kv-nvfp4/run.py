"""Teacher-forced K/V precision runs: one serve engine per --kv, the same prompts, STRATA_LOGPOS top 20 per position.

usage: python run.py [--windows] <out_dir> <strata.exe> <kv>[:<tag>[:ENV=VALUE]] ...
       (the model and pack of setup's strata-iq3_s.json; a fixed --expert-cache 6000 and --adapt-every 100000)

--windows: four texts of ~8-10K tokens, each a single message read and scored entirely through the verify windows
(--short-read 16384). Every --kv then runs the same decode attention kernel in the same order, so the stored K/V is
the only difference (the main measurement).
Without it: a long first message read by the batched prompt path (the tensor-core kernels), a short reply, and a last
message scored through the windows (--short-read 1024); "8k" / "32k" name the first message (10.5-10.7K and
35.8-35.9K prompt tokens in all). That form also measures the prompt kernels' summation order.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PORT = 8093


REV = "296b59d2"   # the texts as they were at this commit (the measurement's base), not the working tree's


def read(path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{REV}:{path}"], capture_output=True, check=True
                          ).stdout.decode("utf-8", errors="replace")


def text_of(paths, chars):
    out = []
    n = 0
    for p in paths:
        t = read(p)
        out.append(t)
        n += len(t)
        if n >= chars:
            break
    return "".join(out)[:chars]


LONG = text_of(["docs/DETAILS.md", "docs/AMD_HIP.md", "docs/MULTI_GPU.md", "src/core/layer.cpp", "src/prefill/prefill.cpp"],
               130000)
TAIL_DOC = read("docs/COMMUNITY_BENCHMARKS.md")[:2400]
TAIL_CODE = read("src/kernels/cuda/kv_stream.cu")[3000:5400]
PROMPTS = {"8k-doc": (LONG[:30000], TAIL_DOC), "32k-doc": (LONG[:120000], TAIL_DOC),
           "8k-code": (LONG[:30000], TAIL_CODE), "32k-code": (LONG[:120000], TAIL_CODE)}
# --windows: one message per text, every token read (and scored) through the verify windows, so every --kv runs the
# same decode attention kernel in the same order and only the stored K/V differs
WINDOWS = "--windows" in sys.argv
if WINDOWS:
    sys.argv.remove("--windows")
    def part(path, a, n=30000):
        return read(path)[a:a + n]
    PROMPTS = {"doc": part("docs/DETAILS.md", 0), "c++": part("src/prefill/prefill.cpp", 20000),
               "python": part("serve/server.py", 20000), "doc2": part("docs/AMD_HIP.md", 0)}


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ready():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    out = Path(sys.argv[1])
    exe = sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((REPO / "strata-iq3_s.json").read_text(encoding="utf-8"))
    for spec in sys.argv[3:]:
        kv, tag, *extra = spec.split(":") + [""]
        tag = tag or kv
        extra_env = dict(e.split("=", 1) for e in extra if e)
        args = list(base["args"])

        def setarg(name, value):
            if name in args:
                i = args.index(name)
                if value is None:
                    del args[i:i + 2]
                else:
                    args[i + 1] = value
            elif value is not None:
                args.extend([name, value])
        setarg("--kv", kv)
        setarg("--max-context", "65536")
        setarg("--kv-resident", None)
        setarg("--expert-cache", "6000")   # fixed, so every run computes the same experts in the same place
        args += ["--short-read", "16384" if WINDOWS else "1024", "--adapt-every", "100000"]
        cfg = dict(base, exe=exe, args=args, log=str(out / f"engine-{tag}.log"))
        cfgp = out / f"cfg-{tag}.json"
        cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        logpos = out / f"logpos-{tag}.tsv"
        if logpos.exists():
            logpos.unlink()
        env = dict(os.environ, STRATA_LOGPOS=str(logpos), STRATA_LOGPOS_TOPK="20", **extra_env)
        srv = subprocess.Popen([sys.executable, str(REPO / "serve/server.py"), "--engine",
                                "strata", "--config", str(cfgp), "--port", str(PORT)], cwd=str(REPO), env=env,
                               stdout=open(out / f"server-{tag}.log", "w"), stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while not ready():
                if srv.poll() is not None:
                    raise SystemExit(f"{tag}: server exited ({srv.returncode})")
                if time.time() - t0 > 1800:
                    raise SystemExit(f"{tag}: not ready after 30 min")
                time.sleep(5)
            print(f"{tag}: ready after {time.time() - t0:.0f} s", flush=True)
            for name, pr in PROMPTS.items():
                if WINDOWS:
                    msgs = [{"role": "user", "content": pr}]
                else:
                    first, tail = pr
                    msgs = [{"role": "user", "content": "Read this.\n\n" + first},
                            {"role": "assistant", "content": "I have read it."},
                            {"role": "user", "content": tail}]
                t1 = time.time()
                r = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0,
                                                  "chat_template_kwargs": {"enable_thinking": False}})
                u = r.get("usage", {})
                print(f"{tag} {name}: prompt_tokens={u.get('prompt_tokens')} {time.time() - t1:.1f} s", flush=True)
                with open(logpos, "a", encoding="utf-8") as f:
                    f.write(f"#END {name}\n")
        finally:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(srv.pid)], capture_output=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
