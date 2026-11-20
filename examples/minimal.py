import pandas as pd
from didsuitepy import didsuite

data = pd.DataFrame([
    {"state": state, "year": year, "g": 3 if state < 3 else 0,
     "y": 0.2*state + 0.3*year + int(state < 3 and year >= 3) + 0.01*state*year}
    for state in range(8) for year in range(1, 6)
])
print(didsuite(data, y="y", time="year", cluster="state", gvar="g",
               estimators=["twfe", "did2s", "bjs"], inference="standard"))
