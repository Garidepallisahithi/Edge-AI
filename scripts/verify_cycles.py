from scipy.io import loadmat
from pathlib import Path

# Load one .mat file (path adjusted relative to scripts folder)
p = next(Path("../data/raw/nasa").rglob("*.mat"))
mat = loadmat(p, squeeze_me=True, struct_as_record=False)

# Get the top-level key (filename, e.g. 'B0031')
top_key = [k for k in mat.keys() if not k.startswith("__")][0]
battery = mat[top_key]
cycle = battery.cycle

# Inspect first few cycles
for i, c in enumerate(cycle[:5]):
    print(f"Cycle {i} type:", c.type)
    print("Ambient temp:", c.ambient_temperature)

    if c.type == "discharge":
        print("Capacity:", c.data.Capacity)
        print("Voltage sample:", c.data.Voltage_measured[:5])
        print("Current sample:", c.data.Current_measured[:5])
        print("Temperature sample:", c.data.Temperature_measured[:5])

    elif c.type == "charge":
        print("Voltage sample:", c.data.Voltage_measured[:5])
        print("Current sample:", c.data.Current_measured[:5])

    elif c.type == "impedance":
        print("Re:", c.data.Re)
        print("Rct:", c.data.Rct)

    print("-" * 40)
