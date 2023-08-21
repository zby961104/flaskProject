import json


def save_result(filename, result):
    file = open(filename, "w", encoding="utf-8")
    json.dump(result, file, ensure_ascii=False)

