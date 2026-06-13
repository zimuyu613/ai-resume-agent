from pathlib import Path

from rag import build_chunk_records, get_local_embedding, split_text


BASE_DIR = Path(__file__).parent


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_split_text() -> None:
    text = "这是第一段简历内容。" * 100
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert_true(len(chunks) > 1, "split_text 应该能把长文本切成多个 chunk")
    assert_true(all(chunk.strip() for chunk in chunks), "split_text 不应该产生空 chunk")


def test_chunk_metadata() -> None:
    text = "Python RAG 项目经历。" * 80
    records = build_chunk_records(text, chunk_size=120, overlap=20)
    assert_true(len(records) > 1, "build_chunk_records 应该能返回多个 chunk record")
    assert_true(all(record["text"].strip() for record in records), "chunk record 的 text 不应该为空")
    assert_true("chunk_id" in records[0], "chunk metadata 应该包含 chunk_id")
    assert_true("chunk_length" in records[0], "chunk metadata 应该包含 chunk_length")


def test_empty_text() -> None:
    chunks = split_text("")
    assert_true(chunks == [], "空文本应该返回空列表")


def test_local_embedding() -> None:
    embedding = get_local_embedding("Python RAG Prompt Engineering")
    assert_true(len(embedding) == 384, "local embedding 默认维度应该是 384")
    assert_true(any(value != 0 for value in embedding), "local embedding 不应该全为 0")


def test_sample_files() -> None:
    resume_path = BASE_DIR / "samples" / "sample_resume.txt"
    job_path = BASE_DIR / "samples" / "sample_job_description.txt"

    assert_true(resume_path.exists(), "sample_resume.txt 应该存在")
    assert_true(job_path.exists(), "sample_job_description.txt 应该存在")
    assert_true(resume_path.read_text(encoding="utf-8").strip(), "sample_resume.txt 不应该为空")
    assert_true(job_path.read_text(encoding="utf-8").strip(), "sample_job_description.txt 不应该为空")


if __name__ == "__main__":
    test_split_text()
    test_chunk_metadata()
    test_empty_text()
    test_local_embedding()
    test_sample_files()
    print("simple_test.py: all tests passed")
