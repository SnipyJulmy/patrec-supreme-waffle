def evaluate_performance(keyword, results, validation_set, transcript, k = 10):
    retrieved = len(results)

    for key, score in results.items():
        print("\tkey : %s \n\tkw : %s" % (transcript[key], keyword))
        if transcript[key] == keyword:
            print("%s : %s" % (key, keyword))

    precision, recall, area_under_curve = 0, 0, 0
    return precision, recall, area_under_curve
