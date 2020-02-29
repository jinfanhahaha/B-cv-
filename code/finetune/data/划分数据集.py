train_data = ""
test_data = ""

with open("data.txt", "r") as f:
    data = f.read().split("\n")

for i, data_i in enumerate(data):
    if i % 8 == 0:
        test_data += data_i + "\n"
    else:
        train_data += data_i + "\n"
train_data = train_data.strip("\n")
test_data = test_data.strip("\n")

with open("train_data.txt", "w") as w:
    w.write(train_data)

with open("test_data.txt", "w") as w:
    w.write(test_data)
