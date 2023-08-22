import json
import datetime
import os.path


def save_result(filename, result):
    file = open(filename, "w", encoding="utf-8")
    json.dump(result, file, ensure_ascii=False)


def saveToJsonFile(filename, data):
    data_list = readJsonFile(filename)
    print("pre: ", data_list)

    data_list.append(data)
    print("after: ", data_list)
    with open(filename, 'w') as f:
        json.dump(data_list, f, ensure_ascii=False)


def readJsonFile(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as load_f:
            data_list = json.load(load_f)
    else:
        data_list = []

    return data_list


if __name__ == "__main__":
    smiles = "CO"
    username = "zby"
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    result = "有活性"
    dataset = "BACE"
    data = {"smiles": smiles, "username": username, "time": time, "result": result, "dataset": dataset}
    filename = "../static/history/single.json"
    saveToJsonFile(filename, data)