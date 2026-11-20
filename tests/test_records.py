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
