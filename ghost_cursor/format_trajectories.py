import json

data_path = "/home/heling/School/code/javascript/samples.json"
with open(data_path, 'r') as f:
    data = json.load(f)
print(len(data))


def format_trajectory(trajectorys):
    lines = []
    for i, trajectory in enumerate(trajectorys):
        id_ = f"{i}"
        data_ = ';'.join(
            [f"{p['x']},{p['y']},{p['t']}" for p in trajectory]
        )
        target_ = f"{trajectory[-1]['x']},{trajectory[-1]['y']}"
        label_ = "0"
        line = " ".join([id_, data_, target_, label_])
        lines.append(line)

    with open("code/data/gc2.csv", "w") as f:
        f.write("\n".join(lines))


format_trajectory(data)