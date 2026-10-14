"""Compare STRATA_LOGPOS top-20 files against a reference run.

usage: python compare.py <dir> <ref_tag> <tag> ...
Per prompt section (#END <name>): argmax agreement with the reference, top-10 overlap, KL(ref || run) over the
reference's top 20 plus one bucket for the rest, and the perplexity of the true tokens; mean +- standard error.
"""
import math
import sys
from pathlib import Path


def load(path):
    sections, rows = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#END"):
            sections[line.split()[1]] = rows
            rows = []
            continue
        f = line.split("\t")
        top = {}
        for kv in f[8:]:
            i, lp = kv.split(":")
            top[int(i)] = float(lp)
        rows.append({"pos": int(f[0]), "tgt_lp": float(f[2]), "argmax": int(f[3]), "top": top})
    return sections


def kl(ref, run):
    pr = {i: math.exp(lp) for i, lp in ref.items()}
    floor = min(run.values())   # a token outside the run's top 20 gets at most its 20th probability
    total, rest_r, rest_q = 0.0, 1.0 - sum(pr.values()), 1.0
    for i, p in pr.items():
        q = math.exp(run.get(i, floor))
        total += p * (math.log(p) - math.log(q))
        rest_q -= q
    rest_r, rest_q = max(rest_r, 1e-12), max(rest_q, 1e-12)
    return total + rest_r * (math.log(rest_r) - math.log(rest_q))


def mean_se(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    return m, math.sqrt(v / len(xs))


def main():
    d = Path(sys.argv[1])
    ref = load(d / f"logpos-{sys.argv[2]}.tsv")
    print("| prompt | KV | positions | argmax same | top-10 overlap | mean KL | perplexity | vs ref (nats/token) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, rrows in ref.items():
        ppl_ref = math.exp(-sum(r["tgt_lp"] for r in rrows) / len(rrows))
        print(f"| {name} | {sys.argv[2]} (ref) | {len(rrows)} | | | | {ppl_ref:.3f} | |")
        for tag in sys.argv[3:]:
            run = load(d / f"logpos-{tag}.tsv")[name]
            if [r["pos"] for r in run] != [r["pos"] for r in rrows]:
                print(f"| {name} | {tag} | positions differ |")
                continue
            same = sum(a["argmax"] == b["argmax"] for a, b in zip(rrows, run)) / len(run)
            ov = sum(len(set(sorted(a["top"], key=a["top"].get, reverse=True)[:10]) &
                         set(sorted(b["top"], key=b["top"].get, reverse=True)[:10])) / 10 for a, b in zip(rrows, run)) / len(run)
            kls = [kl(a["top"], b["top"]) for a, b in zip(rrows, run)]
            dl = [-(b["tgt_lp"] - a["tgt_lp"]) for a, b in zip(rrows, run)]
            km, ks = mean_se(kls)
            dm, ds = mean_se(dl)
            ppl = math.exp(-sum(r["tgt_lp"] for r in run) / len(run))
            print(f"| {name} | {tag} | {len(run)} | {same * 100:.1f}% | {ov * 100:.1f}% | {km:.4f} +- {ks:.4f} | "
                  f"{ppl:.3f} | {dm:+.4f} +- {ds:.4f} |")


if __name__ == "__main__":
    main()
