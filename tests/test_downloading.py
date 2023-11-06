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


def test_download_df_with_all_args(random_df_fixture,tmp_path):

    returned_path = download_df(random_df_fixture, 
                                title="THIS is my.tExt", 
                                save_filepath=tmp_path, 
                                filename_suffix='_test_DF', date=True)

    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    expected_file_path = os.path.join(tmp_path, 
                                      f'ThisIsMyText_test_DF {datetime_str}.csv')

    print("\nexpected: ", expected_file_path)
    print("returned: ", returned_path)
    
    # Failures here won't halt execution
    expect(os.path.exists(expected_file_path), 
           "Expected file not found")
    expect(returned_path==expected_file_path, 
           "Returned value for file path does not match expected value")
    expect(lambda: pd.testing.assert_frame_equal(pd.read_csv(expected_file_path), 
                                                 random_df_fixture, 
                                                 check_dtype=False), 
            "Saved DataFrame does not match input DataFrame")
                          
    # Halt execution and show the stack trace for failed assertion(s)
    assert_expectations() 
    
