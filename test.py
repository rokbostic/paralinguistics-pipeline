from pathlib import Path
import pandas as pd

sed_outputs = Path("outputs") / "output_sed"

for sed_output in sed_outputs.glob("*.csv"):
    df = pd.read_csv(sed_output)
    df = df[df.iloc[:, 0] != "Breathing"]
    if not df.empty:
        events = df.iloc[:, 0].to_list()
        print(sed_output)