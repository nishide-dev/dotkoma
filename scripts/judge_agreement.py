"""Agreement of the machine checks and the VLM judge with review.jsonl labels.

    uv run python scripts/judge_agreement.py configs/judge.toml work/pilot-03 work/pilot-04 ...

Per sample: machine = finalize.jsonl auto_verdict (candidate counts as accept),
judge = the ok judge.jsonl record for the request this judge config makes for the
sample now (same frames, same settings and instruction),
label = review.jsonl verdict (unsure is left out). Samples the machine rejects are
not sent to the judge and are left out here as well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotkoma.config import load_judge_config  # noqa: E402
from dotkoma.judge import build_requests, latest_verdicts  # noqa: E402
from dotkoma.records import latest_by_job, read_jsonl  # noqa: E402


def rows(run: Path, cfg) -> list[dict]:
    fin = {r["job_id"]: r for r in read_jsonl(run / "finalize.jsonl") if r["status"] == "ok"}
    review = latest_by_job(read_jsonl(run / "review.jsonl"))
    verdicts = latest_verdicts(run)
    judged: dict[str, dict] = {}
    for q in build_requests(run, cfg):
        r = verdicts.get(q["request_id"])
        if r and r["status"] == "ok":
            judged[q["job_id"]] = r
    out = []
    for jid, f in fin.items():
        lab = review.get(jid)
        if f["auto_verdict"] == "reject" or not lab or lab["verdict"] == "unsure":
            continue
        if lab["postprocess_id"] != f["postprocess_id"]:
            continue  # the label is for older frames
        j = judged.get(jid)
        out.append(
            {
                "run": run.name,
                "job_id": jid,
                "action": f["requested_action"],
                "subject": f["subject"],
                "label": lab["verdict"] == "accept",
                "label_motion": lab["motion_matches"],
                "label_drawing": lab["drawing_kept"],
                "machine": f["auto_verdict"] == "candidate",
                "judge": None if j is None else j["verdict"] == "accept",
                "judge_motion": None if j is None else j["motion_matches"],
                "judge_drawing": None if j is None else j["drawing_kept"],
                "reason": "" if j is None else j["reason"],
                "label_reason": lab["reason"],
            }
        )
    return out


def table(name: str, rs: list[dict], pred) -> str:
    tp = sum(1 for r in rs if pred(r) and r["label"])
    fp = sum(1 for r in rs if pred(r) and not r["label"])
    fn = sum(1 for r in rs if not pred(r) and r["label"])
    tn = sum(1 for r in rs if not pred(r) and not r["label"])
    n = len(rs)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return f"| {name} | {tp + fp} | {tp} | {fp} | {fn} | {acc:.0%} | {prec:.0%} | {rec:.0%} |"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("judge_config", type=Path)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--quiet", action="store_true", help="tables only, no disagreement list")
    a = p.parse_args()
    cfg = load_judge_config(a.judge_config)
    rs = [r for run in a.runs for r in rows(run, cfg)]
    missing = [r for r in rs if r["judge"] is None]
    if missing:
        print(f"{len(missing)} labelled samples have no judge verdict; they are left out")
    rs = [r for r in rs if r["judge"] is not None]
    print(f"{len(rs)} samples, label accept {sum(r['label'] for r in rs)}\n")
    print(
        "| | passed | passed & label accept | passed & label reject | missed accept "
        "| agreement | precision | recall |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print(table("machine (candidate)", rs, lambda r: r["machine"]))
    print(table("judge (accept)", rs, lambda r: r["judge"]))
    print(table("machine and judge", rs, lambda r: r["machine"] and r["judge"]))
    for k in ("motion", "drawing"):
        both = [r for r in rs if r[f"label_{k}"] is not None]
        same = sum(1 for r in both if r[f"label_{k}"] == r[f"judge_{k}"])
        print(f"\n{k}: judge agrees with the label on {same}/{len(both)}")
    if a.quiet:
        return
    print("\ndisagreements (judge vs label):")
    for r in rs:
        if r["judge"] != r["label"]:
            print(
                f"  {r['run']:15} {r['action']:8} {r['subject'][:22]:22} "
                f"label={'A' if r['label'] else 'R'} judge={'A' if r['judge'] else 'R'} "
                f"m={r['judge_motion']!s:5} d={r['judge_drawing']!s:5} | {r['reason']} "
                f"| label: {r['label_reason']}"
            )


if __name__ == "__main__":
    main()
