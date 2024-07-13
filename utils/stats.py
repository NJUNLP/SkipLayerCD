from collections import defaultdict

stats = defaultdict(list)

def calculate_stats(contain_data: bool = True):
    d = {}
    for key in stats:
        data = [item for item in stats[key] if item is not None]
        key_len = len(data)
        key_sum = sum(data)
        key_avg = key_sum / key_len
        d[key] = {
            'data': stats[key] if contain_data else None,
            'len': key_len,
            'sum': key_sum,
            'avg': key_avg
        }
    return d