from dotkoma.records import append_jsonl, job_id, latest_by_job, read_jsonl, write_jsonl_atomic


def test_job_id_is_deterministic_and_sensitive():
    a = {"source": {"row_id": 1}, "action": "hop", "seed": 3}
    assert job_id(a) == job_id(dict(reversed(list(a.items()))))
    assert job_id(a) != job_id({**a, "seed": 4})


def test_torn_last_line_is_ignored(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"job_id": "a", "status": "ok"})
    with p.open("a") as f:
        f.write('{"job_id": "b", "sta')  # crash mid-append
    assert [r["job_id"] for r in read_jsonl(p)] == ["a"]


def test_latest_record_wins(tmp_path):
    p = tmp_path / "r.jsonl"
    write_jsonl_atomic(p, [{"job_id": "a", "status": "failed"}, {"job_id": "a", "status": "ok"}])
    assert latest_by_job(read_jsonl(p))["a"]["status"] == "ok"


def test_append_after_torn_line_keeps_log_readable(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"job_id": "a", "status": "ok"})
    with p.open("a") as f:
        f.write('{"job_id": "b", "sta')  # crash mid-append
    append_jsonl(p, {"job_id": "c", "status": "ok"})  # the resumed run
    assert [r["job_id"] for r in read_jsonl(p)] == ["a", "c"]


def test_append_to_file_that_is_only_a_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"job_id": "b"')
    append_jsonl(p, {"job_id": "c"})
    assert [r["job_id"] for r in read_jsonl(p)] == ["c"]


def test_torn_multibyte_tail_does_not_block_resume(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"job_id": "a", "error": "失敗"})
    data = '{"job_id": "b", "error": "失敗"}'.encode()
    with p.open("ab") as f:
        f.write(data[: data.index("失".encode()) + 1])  # cut inside a UTF-8 character
    assert [r["job_id"] for r in read_jsonl(p)] == ["a"]
    append_jsonl(p, {"job_id": "c"})
    assert [r["job_id"] for r in read_jsonl(p)] == ["a", "c"]


def test_reader_and_append_agree_on_unterminated_record(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"job_id": "a", "status": "ok"})
    with p.open("a") as f:
        f.write('{"job_id": "b", "status": "ok"}')  # complete JSON, newline never written
    before = read_jsonl(p)
    append_jsonl(p, {"job_id": "c", "status": "ok"})
    after = read_jsonl(p)
    # "b" was never complete: not trusted before the append, not silently lost after it
    assert [r["job_id"] for r in before] == ["a"]
    assert [r["job_id"] for r in after] == ["a", "c"]


def test_hand_edited_file_without_final_newline(tmp_path):
    p = tmp_path / "selections.jsonl"
    p.write_text('{"job_id": "a", "start": 0}\n{"job_id": "b", "start": 2}')
    assert [r["job_id"] for r in read_jsonl(p, unterminated_tail="parse")] == ["a", "b"]


def test_corrupt_complete_line_raises(tmp_path):
    import pytest

    p = tmp_path / "r.jsonl"
    p.write_bytes(b'{"job_id": "a"}\nnot json\n{"job_id": "c"}\n')
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(p)
