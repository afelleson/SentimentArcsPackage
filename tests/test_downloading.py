'''test_downloading.py

Pytest-formatted tests for downloading-related functions defined in 
simplifiedSA.py.
'''

import os
import datetime
import random
import string

import pandas as pd
import numpy as np

import pytest
from delayed_assert import delayed_assert, expect, assert_expectations

from imppkg.simplifiedSA import uniquify, download_df

@pytest.fixture()
def random_df_fixture():
    """Create a pd.DataFrame populated with random data of various types.
    """
    random_df = pd.DataFrame({'Letter':["A", "B", "C", "D"],
                            'Index': [0, 1, 2, 3],
                            'Score':[0.234, 0.948, 0.23, 0],
                           })

    num_rows = 10
    
    # Generate random strings for col1
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))
    col1 = [random_string(5) for _ in range(num_rows)]
    
    # Generate random integers for col2
    col2 = np.random.randint(1, 10, num_rows)

    # Generate random doubles for col3
    col3 = np.random.uniform(1.0, 10.0, num_rows)

    # Create a DataFrame
    df = pd.DataFrame({
        "col1": col1,
        "col2": col2,
        "col3": col3
    })
    return random_df

@pytest.mark.parametrize(
    "kwargs, expected_file_name",
    [
        ({"filename_suffix":'_test_DF', "date":True}, 
         f"ThisIsMyText_test_DF {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.csv"), 
        ({"filename_suffix":"", "date":True}, 
         f"ThisIsMyText {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.csv"), 
        ({"date":True}, 
         f"ThisIsMyText_df {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.csv"), 
        ({"filename_suffix":"_test_DF", "date":False}, 
         "ThisIsMyText_test_DF.csv"), 
        ({}, 
         "ThisIsMyText_df.csv"),
    ],
    ids=["all_args", "empty_suffix", "default_suffix", "no_date", "default_suffix_no_date"],
)
def test_download_df(random_df_fixture,tmp_path, kwargs, expected_file_name):
    """Note: Sometimes may fail when it shouldn't due to a minute
    changeover in the datetime string if run at just the wrong time.
    """

    returned_path = download_df(random_df_fixture, 
                                title="THIS is my.tExt", 
                                save_filepath=tmp_path, 
                                **kwargs)

    expected_path = os.path.join(tmp_path, expected_file_name)
    
    expect(returned_path==expected_path, 
           "Returned value for file path does not match expected value:" +
           "\n-- Expected: " + expected_path + 
           "\n-- Returned: " + returned_path)
    expect(os.path.exists(expected_path), 
           "Expected file not found")
    expect(lambda: pd.testing.assert_frame_equal(pd.read_csv(expected_path), 
                                                 random_df_fixture, 
                                                 check_dtype=False), 
            "Saved DataFrame does not match input DataFrame")
    assert_expectations()
    
def test_download_df_TypeError(tmp_path):
    array = [["a","b","c"],[1,2,3]]
    
    with pytest.raises(TypeError) as error_info:
        download_df(array, title="text title", save_filepath=tmp_path)
        
    assert "list" in str(error_info.value)
    
    
    