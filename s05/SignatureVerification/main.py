import numpy as np
from dtw import DTW
import itertools
import operator
import csv

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def get_users_enrollment(users):
    users_enrollment = {}
    for user in users:
        for i in range(1,6):
            value = str(i)
            if i < 10:
                value = '0'+str(i)
            user_data = np.loadtxt('enrollment/'+user+'-g-'+value+'.txt')
            if user not in users_enrollment:
                users_enrollment[user] = []
            users_enrollment[user].append(user_data)
    return users_enrollment

def get_users_enrollment_features(users_enrollment):
    users_enrollment_features = {}
    for user,signatures in users_enrollment.items():
        users_enrollment_features[user] = []
        for signature in signatures:
            features = get_features(signature)
            users_enrollment_features[user].append(features)
    return users_enrollment_features

def get_ground_truth(filename):
    ground_truth = {}
    gt = np.loadtxt(filename,dtype=str)
    for sign in gt:
        infos = sign[0].split('-')
        if infos[0] not in ground_truth:
            ground_truth[infos[0]] = {}
        result = False
        if sign[1] == 'g':
            result = True
        ground_truth[infos[0]][infos[1]] = result
    return ground_truth

def get_users_verification(users):
    users_verification = {}
    for user in users:
        for i in range(1,46):
            value = str(i)
            if i < 10:
                value = '0'+str(i)
            file = user+'-'+value
            user_data = np.loadtxt('verification/'+file+'.txt')
            if user not in users_verification:
                users_verification[user] = []
            users_verification[user].append(user_data)
    return users_verification

def get_users_verification_features(users_verification):
    users_verification_features = {}
    for user,signatures in users_verification.items():
        users_verification_features[user] = []
        for signature in signatures:
            features = get_features(signature)
            users_verification_features[user].append(features)
    return users_verification_features

def get_features(signature):
    features = []
    i = 0
    for sign in signature:
        x = sign[1]
        y = sign[2]
        pressure = sign[3]
        if i == 0:
            vx = 0
            vy = 0
        else:
            vx = x - signature[i-1][1]
            vy = y - signature[i-1][2]
        feature = np.array([x, y, vx, vy, pressure])
        features.append(feature)
        i =+ 1
    return scale_linear_bycolumn(features)
def get_users_mean_enrollment_scores(users_enrollment_features):
    users_mean_scores = {}
    for user in users_enrollment_features:
        print(user)
        scores = []
        dtw = None
        for a, b in itertools.combinations(users_enrollment_features[user], 2):
            if dtw == None:
                dtw = DTW(a)
            if not np.array_equal(a,dtw.features):
                dtw = DTW(a)
            scores.append(dtw.calculate_cost(b))
        users_mean_scores[user] = np.mean(scores)
    return users_mean_scores

def get_costs_normalize_user(user):
    cost = np.load('costs/'+user+'.npy')
    cost = cost.item()
    vals = np.fromiter(cost.values(), dtype=float)
    costs_norm = scale_linear_bycolumn(vals)
    i = 0
    for key, value in cost.items():
        cost[key] = costs_norm[i]
        i+=1
    return cost

def get_costs_normalize(users):
    result = {}
    for user in users:
        cost = get_costs_normalize_user(user)
        result = {**result, **cost}
    return result

def find_best_threshold(users, users_enrollment_features, users_verification_features, users_mean_enrollment_scores,  ground_truth):
    # results = {}
    # for user in users: 
    #     i = 1
    #     print("Currently using DTW on user "+user)
    #     for features in users_verification_features[user]:
    #         dtw = DTW(features)
    #         costs = []
    #         for base_features in users_enrollment_features[user]:
    #             cost = dtw.calculate_cost(base_features)
    #             costs.append(cost)
    #         value = str(i)
    #         if i < 10:
    #             value = '0'+str(i)
    #         file = user+'-'+value
    #         results[file] = np.mean(costs)
    #         i += 1
    results = get_costs_normalize(users)
    thresholds = [5,10,20,30,40,50]
    for threshold in thresholds:
        number_correct = 0
        for key,value in results.items():
            infos = key.split('-')
            real_result = ground_truth[infos[0]][infos[1]]
            result = False
            new_value = value
            if new_value <= threshold:
                result = True 
            if real_result == result:
                number_correct += 1
        print("Accuracy for "+str(threshold)+": ", (number_correct/len(results)))
        # best = 2500

def compute_dtw_and_save(users, users_enrollment_features, users_verification_features, users_mean_enrollment_scores,  ground_truth):
    for user in users: 
        results = {}
        i = 1
        print("Currently using DTW on user "+user)
        length = len(users_verification_features[user])
        for features in users_verification_features[user]:
            # print(i,"/",length)
            dtw = DTW(features)
            costs = []
            for base_features in users_enrollment_features[user]:
                cost = dtw.calculate_cost(base_features)
                costs.append(cost)
            value = str(i)
            if i < 10:
                value = '0'+str(i)
            file = user+'-'+value
            results[file] = np.mean(costs)
            i += 1
        np.save('costs/'+user, results)

def write_to_csv(users):
    output_lines = []
    for user in users:
        output_line = [user]
        cost = get_costs_normalize_user(user)
        cost = dict(sorted(cost.items(), key=operator.itemgetter(1)))
        for key, value in cost.items():
            output_line.append(key)
            output_line.append(value)
        output_lines.append(output_line)
    with open("output.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_lines)

def main():
    users = np.loadtxt('users.txt',dtype=str)
    
    users_enrollment = get_users_enrollment(users)
    users_enrollment_features = get_users_enrollment_features(users_enrollment)
    #users_mean_enrollment_scores = get_users_mean_enrollment_scores(users_enrollment_features)
    #np.save('users_mean_enrollment_scores',users_mean_enrollment_scores)
    users_mean_enrollment_scores = np.load('users_mean_enrollment_scores.npy')
    users_mean_enrollment_scores = users_mean_enrollment_scores.item()
    users_verification = get_users_verification(users)
    users_verification_features = get_users_verification_features(users_verification)
    ground_truth = get_ground_truth('gt.txt')
    #compute_dtw_and_save(users, users_enrollment_features, users_verification_features, users_mean_enrollment_scores, ground_truth)
    #find_best_threshold(users, users_enrollment_features, users_verification_features, users_mean_enrollment_scores, ground_truth)
    write_to_csv(users)
    

main()