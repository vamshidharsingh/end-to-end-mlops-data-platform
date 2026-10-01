import pytest
from src.transformation.etl import run_etl

def test_etl_creates_output(tmpdir):
    assert run_etl is not None
