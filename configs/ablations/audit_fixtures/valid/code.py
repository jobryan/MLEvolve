import pandas as pd

submission = pd.DataFrame({"id": [1, 2], "prediction": [0.2, 0.8]})
submission.to_csv("/home/submission/submission.csv", index=False)
