"""prepare (local CPU): pick sprites, build reference images, write jobs.jsonl."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .imaging import Layout, choose_key_color, make_reference, save_png
from .pixel_ops import expand as expand_pixel_ops
from .pixel_ops import fits, sprite_digest
from .plan import latest_plans, planner_spec, request_id
from .records import derive_seed, job_id, stable_hash, write_json_atomic, write_jsonl_atomic
from .source import Sprite, load_sprites, source_ref

log = logging.getLogger(__name__)


def assign_split(subject: str, cfg: Config) -> str:
    """Split by subject so variants of one subject never straddle train/eval."""
    u = int(stable_hash([cfg.split.salt, subject], 8), 16) / 0xFFFFFFFF
    if u < cfg.split.test:
        return "test"
    if u < cfg.split.test + cfg.split.val:
        return "val"
    return "train"


def _shape_distance(a: Sprite, b: Sprite) -> float:
    """1 - IoU of alpha masks: recolours of the same shape score ~0."""
    ma, mb = a.rgba[..., 3] > 0, b.rgba[..., 3] > 0
    union = (ma | mb).sum()
    return 1.0 - ((ma & mb).sum() / union) if union else 0.0


def pick_distinct(candidates: list[Sprite], count: int, min_distance: float = 0.15) -> list[Sprite]:
    """Greedy pick of sprites that differ in silhouette, not just in colour."""
    chosen: list[Sprite] = []
    for c in candidates:
        if len(chosen) >= count:
            break
        if all(_shape_distance(c, p) >= min_distance for p in chosen):
            chosen.append(c)
    return chosen


def _sprite_groups(pick, pool: list[Sprite], cfg: Config):
    """(candidate sprites, actions) to pick sprites for, separately.

    Video actions share the pick's pool in source order (so their job_ids are as
    before). Each pixel_ops action picks on its own among the sprites with room
    for its motion: most Atlas sprites fill the 16x16 grid, and a translation
    would cut them off. A pinned row_id is kept as is (prepare then skips the
    motion if it does not fit).
    """
    video = tuple(a for a in pick.actions if cfg.actions[a].generation_method == "video_model")
    if video:
        yield pool, video
    for a in pick.actions:
        action = cfg.actions[a]
        if action.generation_method != "pixel_ops":
            continue
        if pick.row_id is None:
            offsets = expand_pixel_ops(action.pixel_ops)["offsets"]
            roomy = [sp for sp in pool if fits(sp.rgba, offsets)]
            if not roomy:
                log.warning("%s: no sprite has room for %s %s", pick.subject, a, offsets)
                continue
            yield roomy, (a,)
        else:
            yield pool, (a,)


def model_spec(g) -> dict:
    """Everything about the model that changes its output; part of job_id.

    Optional parts are omitted when unused so existing job_ids stay valid.
    """
    spec = {"backend": g.backend, "id": g.model_id, "revision": g.model_revision}
    if g.loras:
        spec["loras"] = [
            {
                "repo": x.repo,
                "file": x.file,
                "revision": x.revision,
                "target": x.target,
                "weight": x.weight,
            }
            for x in g.loras
        ]
    if g.quantization != "none":
        spec["quantization"] = g.quantization
    if not g.vae_tiling:
        spec["vae_decode"] = "full"
    return spec


def build_prompt(template: str, sprite: Sprite) -> str:
    return template.format(subject=sprite.subject, caption=sprite.caption)


def prepare(cfg: Config) -> Path:
    run = cfg.run_dir
    inputs = run / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    subjects = {p.subject for p in cfg.sprites}
    found = load_sprites(cfg.source, subjects)
    missing = sorted(subjects - set(found))
    if missing:
        raise SystemExit(f"subjects not found in source: {missing}")

    g, r = cfg.generation, cfg.reference
    layout = Layout.centered(g.width, g.height, r.scale)
    src = source_ref(cfg.source)

    jobs: list[dict] = []
    seen: set[str] = set()
    plans = _PlanBook(run, cfg)
    for pick in cfg.sprites:
        pool = found[pick.subject]
        if pick.row_id is not None:
            pool = [s for s in pool if s.row_id == pick.row_id]
            if not pool:
                raise SystemExit(f"row_id {pick.row_id} ({pick.subject}) not found")
        for group_pool, group_actions in _sprite_groups(pick, pool, cfg):
            sprites = pick_distinct(group_pool, pick.count)
            if len(sprites) < pick.count:
                log.warning(
                    "%s %s: only %d distinct sprites (wanted %d)",
                    pick.subject,
                    list(group_actions),
                    len(sprites),
                    pick.count,
                )

            for sp in sprites:
                key = choose_key_color(sp.rgba, r.key_colors)
                stem = f"{sp.row_id:07d}"
                src_png = inputs / f"{stem}.png"
                ref_png = inputs / f"{stem}_ref_{key.lstrip('#')}.png"
                save_png(src_png, sp.rgba)
                if any(cfg.actions[a].generation_method == "video_model" for a in group_actions):
                    save_png(ref_png, make_reference(sp.rgba, layout, key))

                for action_name, variant in (
                    (a, v)
                    for a in group_actions
                    for v in cfg.actions[a].variant_indices(pick.prompt_variants)
                ):
                    action = cfg.actions[action_name]
                    if action.generation_method == "pixel_ops":
                        job = _pixel_ops_job(cfg, sp, action, src, src_png, run)
                        if job and job["job_id"] not in seen:
                            seen.add(job["job_id"])
                            jobs.append(job)
                        continue
                    plan_info = None
                    if action.method == "vlm":
                        prompt, plan_info = _planned_prompt(
                            cfg, sp, action, src, src_png, run, plans
                        )
                        if prompt is None:
                            continue  # awaiting a plan, or the VLM found the motion infeasible
                    else:
                        prompt = build_prompt(action.prompts[variant], sp)
                    # same seed for every wording, so variants differ only in the prompt
                    seed = (
                        g.seed if g.seed is not None else derive_seed(src, sp.row_id, action_name)
                    )
                    gen_inputs = {
                        "source": {**src, "row_id": sp.row_id},
                        "action": action_name,
                        "prompt": prompt,
                        "negative_prompt": g.negative_prompt,
                        "seed": seed,
                        "model": model_spec(g),
                        "inference": {
                            "width": g.width,
                            "height": g.height,
                            "num_frames": g.num_frames,
                            "fps": g.fps,
                            "steps": g.steps,
                            "guidance": g.guidance,
                            **({"guidance_2": g.guidance_2} if g.guidance_2 is not None else {}),
                        },
                        "reference": {"layout": layout.to_dict(), "key_color": key},
                    }
                    if plan_info:
                        # new kind of job: tie it to the exact pixels (template jobs keep
                        # their existing ids)
                        gen_inputs["sprite_sha256"] = sprite_digest(sp.rgba)
                    jid = job_id(gen_inputs)
                    if jid in seen:
                        continue
                    seen.add(jid)
                    jobs.append(
                        {
                            "job_id": jid,
                            "config_version": cfg.version,
                            "source": {
                                **src,
                                "row_id": sp.row_id,
                                "subject": sp.subject,
                                "caption": sp.caption,
                            },
                            "split": assign_split(sp.subject, cfg),
                            "requested_action": action_name,
                            "motion": action.motion,
                            "generation_method": "video_model",
                            "prompt_method": action.method,
                            "prompt_variant": variant,
                            "prompt": prompt,
                            "negative_prompt": g.negative_prompt,
                            "seed": seed,
                            "model": gen_inputs["model"],
                            "inference": gen_inputs["inference"],
                            "layout": layout.to_dict(),
                            "key_color": key,
                            "input_png": src_png.relative_to(run).as_posix(),
                            "reference_png": ref_png.relative_to(run).as_posix(),
                            **({"plan": plan_info} if plan_info else {}),
                        }
                    )

    jobs.sort(
        key=lambda j: (j["source"]["row_id"], j["requested_action"], j["prompt_variant"] or 0)
    )
    jobs_path = run / "jobs.jsonl"
    write_jsonl_atomic(jobs_path, jobs)
    plans.finish()
    write_json_atomic(run / "config.snapshot.json", cfg.to_dict())
    log.info("wrote %d jobs to %s", len(jobs), jobs_path)
    _report_coverage(jobs)
    return jobs_path


class _PlanBook:
    """Plan requests prepare asks for, and the answers plan.py has written so far."""

    def __init__(self, run: Path, cfg: Config) -> None:
        self.run = run
        self.spec = planner_spec(cfg.plan) if cfg.plan else None
        self.answers = latest_plans(run)
        self.requests: dict[str, dict] = {}
        self.counts = {"planned": 0, "infeasible": 0, "awaiting": 0}

    def finish(self) -> None:
        if self.spec is None:
            return
        reqs = sorted(self.requests.values(), key=lambda q: q["request_id"])
        write_jsonl_atomic(self.run / "plan_requests.jsonl", reqs)
        log.info("plans: %s", self.counts)
        if self.counts["awaiting"]:
            log.warning(
                "%d (sprite, action) pairs await a plan: run `plan`, then prepare again",
                self.counts["awaiting"],
            )


def _planned_prompt(cfg: Config, sp: Sprite, action, src: dict, src_png: Path, run: Path, book):
    """(prompt, plan info) from the VLM's plan, or (None, None) if there is none to use."""
    source = {**src, "row_id": sp.row_id}
    act = {"name": action.name, "description": action.description}
    instruction = book.spec["instruction"].format(
        subject=sp.subject,
        caption=sp.caption,
        action=action.name,
        description=action.description,
    )
    digest = sprite_digest(sp.rgba)
    rid = request_id(source, act, book.spec, instruction, digest)
    book.requests[rid] = {
        "request_id": rid,
        "source": {**source, "subject": sp.subject, "caption": sp.caption},
        "action": act,
        "input_png": src_png.relative_to(run).as_posix(),
        "sprite_sha256": digest,
        "planner": book.spec,
        "instruction": instruction,
    }
    ans = book.answers.get(rid)
    if not ans or ans["status"] != "ok":
        book.counts["awaiting"] += 1
        return None, None
    if not ans["feasible"]:
        book.counts["infeasible"] += 1
        log.info("%s row %d: no %s (%s)", sp.subject, sp.row_id, action.name, ans["reason"])
        return None, None
    book.counts["planned"] += 1
    prompt = action.prompt_frame.format(
        subject=sp.subject, caption=sp.caption, motion_text=ans["motion_text"]
    )
    return prompt, {"request_id": rid, "moving_part": ans["moving_part"]}


def _pixel_ops_job(cfg: Config, sp: Sprite, action, src: dict, src_png: Path, run: Path):
    """A job made by translating the sprite: no prompt, model or reference image."""
    ops = expand_pixel_ops(action.pixel_ops)
    if not fits(sp.rgba, ops["offsets"]):
        # moving it would push pixels off the 16x16 grid: no valid sample to make
        log.warning(
            "%s row %d (pinned): no room for %s (%s); skipped",
            sp.subject,
            sp.row_id,
            action.name,
            ops["offsets"],
        )
        return None
    digest = sprite_digest(sp.rgba)
    jid = job_id(
        {
            "source": {**src, "row_id": sp.row_id},
            "action": action.name,
            "generation_method": "pixel_ops",
            "pixel_ops": ops,
            # the frames are made from these exact pixels, so an edited sprite is a new job
            "sprite_sha256": digest,
        }
    )
    return {
        "job_id": jid,
        "config_version": cfg.version,
        "source": {**src, "row_id": sp.row_id, "subject": sp.subject, "caption": sp.caption},
        "split": assign_split(sp.subject, cfg),
        "requested_action": action.name,
        "motion": action.motion,
        "generation_method": "pixel_ops",
        "prompt_method": None,
        "prompt_variant": None,
        "pixel_ops": ops,
        "sprite_sha256": digest,  # finalize refuses an input_png with other pixels
        "input_png": src_png.relative_to(run).as_posix(),
    }


def _report_coverage(jobs: list[dict]) -> None:
    subjects = sorted({j["source"]["subject"] for j in jobs})
    actions = sorted({j["requested_action"] for j in jobs})
    per_action = {
        a: len({j["source"]["subject"] for j in jobs if j["requested_action"] == a})
        for a in actions
    }
    log.info(
        "subjects=%d actions=%d subjects-per-action=%s", len(subjects), len(actions), per_action
    )
    lone = [a for a, n in per_action.items() if n < 2]
    if lone:
        log.warning(
            "actions used by a single subject (label can be learned from subject): %s", lone
        )
