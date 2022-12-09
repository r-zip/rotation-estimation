import json

import matplotlib.pyplot as plt

with open("history1.json", "r") as f:
    contents = json.load(f)


plt.plot(contents["train"]["mse"])
plt.plot(contents["val"]["mse"])
plt.show()
