import os
import pytest
import pandas as pd
from src.data_loader import load_data

def test_load_data_success(tmp_path):
    # Create a temporary CSV file
    data = "col1,col2\n1,2\n3,4"
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text(data)
    
    # Test loading the CSV file
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert df.shape[1] == 2

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
