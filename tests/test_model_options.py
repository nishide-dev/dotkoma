"""LoRA merging and fp8 storage, checked against a recording fake pipeline (no GPU)."""

import dataclasses
from types import SimpleNamespace

import pytest

from dotkoma.config import LoraConfig
from dotkoma.generate import apply_model_options
from dotkoma.prepare import prepare
from dotkoma.records import read_jsonl

FAKE_TORCH = SimpleNamespace(float8_e4m3fn="fp8", bfloat16="bf16")


class Module:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def enable_layerwise_casting(self, storage_dtype, compute_dtype):
        self.log.append(("cast", self.name, storage_dtype, compute_dtype))


class Pipe:
    def __init__(self, two_experts=True):
        self.log = []
        self.transformer = Module("transformer", self.log)
        self.transformer_2 = Module("transformer_2", self.log) if two_experts else None

    def load_lora_weights(self, repo, weight_name, revision, adapter_name, **kw):
        target = "transformer_2" if kw.get("load_into_transformer_2") else "transformer"
        self.log.append(("load", adapter_name, target, weight_name, revision))

    def set_adapters(self, names, adapter_weights):
        self.log.append(("set", tuple(names), tuple(adapter_weights)))

    def fuse_lora(self, components, adapter_names, lora_scale):
        self.log.append(("fuse", tuple(components), tuple(adapter_names)))

    def unload_lora_weights(self):
        self.log.append(("unload",))


LORAS = [
    {
        "repo": "r",
        "file": "high.safetensors",
        "revision": "a" * 40,
        "target": "transformer",
        "weight": 1.0,
    },
    {
        "repo": "r",
        "file": "low.safetensors",
        "revision": "a" * 40,
        "target": "transformer_2",
        "weight": 0.5,
    },
]


def test_loras_go_to_their_expert_and_are_fused_then_unloaded():
    pipe = Pipe()
    apply_model_options(pipe, {"loras": LORAS}, FAKE_TORCH)
    kinds = [e[0] for e in pipe.log]
    assert kinds == ["load", "load", "set", "fuse", "set", "fuse", "unload"]
    assert ("load", "lora0", "transformer", "high.safetensors", "a" * 40) in pipe.log
    assert ("load", "lora1", "transformer_2", "low.safetensors", "a" * 40) in pipe.log
    assert ("set", ("lora0",), (1.0,)) in pipe.log and (
        "fuse",
        ("transformer",),
        ("lora0",),
    ) in pipe.log
    assert ("set", ("lora1",), (0.5,)) in pipe.log
    assert ("fuse", ("transformer_2",), ("lora1",)) in pipe.log
    assert not any(e[0] == "cast" for e in pipe.log)  # no quantization requested


def test_fp8_storage_is_applied_after_fusing_to_every_expert():
    pipe = Pipe()
    apply_model_options(pipe, {"loras": LORAS, "quantization": "fp8_storage"}, FAKE_TORCH)
    unload = pipe.log.index(("unload",))
    casts = [i for i, e in enumerate(pipe.log) if e[0] == "cast"]
    assert len(casts) == 2 and min(casts) > unload
    assert ("cast", "transformer", "fp8", "bf16") in pipe.log
    assert ("cast", "transformer_2", "fp8", "bf16") in pipe.log


def test_single_expert_model_without_options_is_untouched():
    pipe = Pipe(two_experts=False)
    apply_model_options(pipe, {"quantization": "fp8_storage"}, FAKE_TORCH)
    assert pipe.log == [("cast", "transformer", "fp8", "bf16")]
    pipe = Pipe(two_experts=False)
    apply_model_options(pipe, {}, FAKE_TORCH)
    assert pipe.log == []


def _ids(cfg):
    return {
        (j["source"]["row_id"], j["requested_action"]): j["job_id"]
        for j in read_jsonl(prepare(cfg))
    }


@pytest.mark.parametrize(
    "change",
    [
        {"loras": (LoraConfig("r", "high.safetensors", "a" * 40),)},
        {"loras": (LoraConfig("r", "high.safetensors", "a" * 40, weight=0.8),)},
        {"quantization": "fp8_storage"},
        {"guidance_2": 1.0},
    ],
)
def test_output_changing_options_change_every_job_id(smoke_cfg, change):
    base = _ids(smoke_cfg)
    changed = _ids(
        dataclasses.replace(
            smoke_cfg, generation=dataclasses.replace(smoke_cfg.generation, **change)
        )
    )
    assert base.keys() == changed.keys()
    assert all(base[k] != changed[k] for k in base)


def test_defaults_keep_the_model_spec_unchanged(smoke_cfg):
    """Unused options are omitted, so configs without them keep their job_ids."""
    for j in read_jsonl(prepare(smoke_cfg)):
        assert set(j["model"]) == {"backend", "id", "revision"}
        assert "guidance_2" not in j["inference"]


def test_offload_modes():
    import pytest

    from dotkoma.generate import offload_mode

    assert offload_mode(True) == offload_mode("model") == "model"
    assert offload_mode(False) == offload_mode("none") == "none"
    assert offload_mode("resident") == "resident"
    with pytest.raises(ValueError):
        offload_mode("sometimes")


def test_offload_is_validated(smoke_cfg):
    import pytest

    from dotkoma.config import ConfigError, load_config

    path = smoke_cfg.config_path
    text = path.read_text()
    path.write_text(text.replace('backend = "fake"', 'backend = "fake"\noffload = "resident"'))
    assert load_config(path).generation.offload == "resident"
    path.write_text(text.replace('backend = "fake"', 'backend = "fake"\noffload = "maybe"'))
    with pytest.raises(ConfigError):
        load_config(path)


class _Mod:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class _FakeTorch:
    class cuda:  # noqa: N801
        @staticmethod
        def empty_cache():
            pass

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def inference_mode():
        import contextlib

        return contextlib.nullcontext()


class _FakeWanPipe:
    def __init__(self):
        self.text_encoder, self.transformer, self.transformer_2, self.vae = (
            _Mod(),
            _Mod(),
            _Mod(),
            _Mod(),
        )
        self.encoded = []

    def __call__(self, image=None, prompt=None, max_sequence_length=512):
        pass

    def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance, **kw):
        assert self.text_encoder.device == "cuda"  # one encoding path: on the GPU
        assert self.transformer.device == "cpu"  # before the experts move in
        self.encoded.append((prompt, kw["max_sequence_length"], kw["device"]))

        class T:
            def cpu(self):
                return self

        return T(), T()


def _resident_backend():
    from dotkoma.generate import DiffusersWanBackend

    b = DiffusersWanBackend.__new__(DiffusersWanBackend)
    b.pipe, b.torch, b.mode, b.resident, b.prompt_cache, b.info = (
        _FakeWanPipe(),
        _FakeTorch(),
        "resident",
        False,
        {},
        {},
    )
    return b


def test_resident_prepare_encodes_once_with_the_pipeline_length(monkeypatch):
    import pytest

    import dotkoma.generate as gen

    monkeypatch.setattr(gen, "_trim", lambda emb, torch: emb)  # covered separately below
    b = _resident_backend()
    pipe = b.pipe
    jobs = [{"prompt": p, "negative_prompt": ""} for p in ("a", "b", "a")]
    b.prepare(jobs)
    assert pipe.encoded == [("a", 512, "cuda"), ("b", 512, "cuda")]
    assert b.resident and pipe.text_encoder is None
    assert pipe.transformer.device == pipe.transformer_2.device == pipe.vae.device == "cuda"
    b.prepare(jobs[:1])  # known prompts: fine
    with pytest.raises(RuntimeError, match="after the experts became resident"):
        b.prepare([{"prompt": "new", "negative_prompt": ""}])
    monkeypatch.setattr(gen, "MAX_RESIDENT_PROMPTS", 1)
    with pytest.raises(RuntimeError, match="split the jobs"):
        _resident_backend().prepare(jobs)


def test_resident_placement_is_part_of_job_id(smoke_cfg):
    import pytest

    from dotkoma.config import load_config
    from dotkoma.generate import load_backend
    from dotkoma.prepare import model_spec, prepare
    from dotkoma.records import read_jsonl

    before = {j["job_id"] for j in read_jsonl(prepare(smoke_cfg))}
    path = smoke_cfg.config_path
    text = path.read_text()
    for value in ("true", "false", '"model"', '"none"'):  # runtime choices: same ids
        path.write_text(text.replace('backend = "fake"', f'backend = "fake"\noffload = {value}'))
        cfg = load_config(path, work_dir=smoke_cfg.work_dir)
        assert {j["job_id"] for j in read_jsonl(prepare(cfg))} == before
    path.write_text(text.replace('backend = "fake"', 'backend = "fake"\noffload = "resident"'))
    cfg = load_config(path, work_dir=smoke_cfg.work_dir)
    assert model_spec(cfg.generation)["placement"] == "resident"
    assert {j["job_id"] for j in read_jsonl(prepare(cfg))}.isdisjoint(before)
    with pytest.raises(ValueError, match="config"):
        load_backend({"backend": "diffusers", "id": "x", "revision": "y"}, "resident")


def test_trimmed_embeddings_restore_exactly():
    import numpy as np

    class T:  # the few torch ops _trim/_padded use, on numpy
        def __init__(self, a):
            self.a = a
            self.shape = a.shape

        def __getitem__(self, k):
            return T(self.a[k])

        def __ne__(self, v):
            return T(self.a != v)

        def any(self, dim):
            return T(self.a.any(axis=dim))

        def nonzero(self):
            return T(np.nonzero(self.a)[0])

        def __len__(self):
            return len(self.a)

        def __int__(self):
            return int(self.a)

        def clone(self):
            return T(self.a.copy())

        def new_zeros(self, *shape):
            return T(np.zeros(shape, self.a.dtype))

    class FakeTorch:
        @staticmethod
        def cat(xs, dim):
            return T(np.concatenate([x.a for x in xs], axis=dim))

    from dotkoma.generate import _padded, _trim

    emb = np.zeros((1, 512, 8), np.float32)
    emb[0, :37] = np.random.default_rng(0).normal(size=(37, 8))
    trimmed = _trim(T(emb), FakeTorch)
    assert trimmed.shape == (1, 37, 8)
    np.testing.assert_array_equal(_padded(trimmed, 512, FakeTorch).a, emb)
