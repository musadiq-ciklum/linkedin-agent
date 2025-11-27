import pytest
from pathlib import Path
from src.data_prep.pipeline import prepare_data
from src.data_prep.saver import save_chunks

@pytest.fixture
def prepared_chunks():
    input_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample.txt"
    return prepare_data(input_path)

def test_prepare_data_runs(prepared_chunks):
    assert isinstance(prepared_chunks, list)
    assert len(prepared_chunks) > 0

def test_save_chunks(prepared_chunks, tmp_path):
    output_path = tmp_path / "sample.json"

    save_chunks(prepared_chunks, output_path)

    assert output_path.exists()
    assert output_path.read_text() != ""

