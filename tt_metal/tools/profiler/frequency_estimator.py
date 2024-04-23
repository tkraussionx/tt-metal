from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.datasets import make_regression

Xs = []
ys = []
with open(
    "/home/mmemarian/models/tt-metal/generated/profiler/.logs/sync_device_info.csv", "r", encoding="utf-8-sig"
) as csvFile:
    # with open ("/home/mmemarian/models/tt-metal/sample_freq_no_tracy_time.csv" , "r", encoding='utf-8-sig') as csvFile:
    for line in csvFile.readlines():
        vals = line.split(",")
        if vals[0] == "device":
            Xs.append([])
            ys.append([])
        else:
            xVal = float(vals[2].strip())
            yVal = int(vals[0].strip())
            Xs[-1].append([xVal])
            ys[-1].append(yVal)

for X, y in zip(Xs, ys):
    # print (X)
    # print (y)
    reg = TheilSenRegressor(max_subpopulation=1e5, max_iter=3000, random_state=5).fit(X, y)
    print(reg.score(X, y))
    print(reg.coef_[0])
    print(reg.__dict__)
