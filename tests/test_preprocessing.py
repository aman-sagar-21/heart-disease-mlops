import pandas as pd

def test_dataframe_columns():

    sample = pd.DataFrame({
        "age":[60],
        "sex":[1],
        "cp":[2],
        "trestbps":[120],
        "chol":[220],
        "fbs":[0],
        "restecg":[1],
        "thalach":[150],
        "exang":[0],
        "oldpeak":[1.5],
        "slope":[2],
        "ca":[0],
        "thal":[3]
    })

    assert sample.shape[1] == 13