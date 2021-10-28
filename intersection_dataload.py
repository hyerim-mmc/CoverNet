import os
import pickle
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

def intersection_data_save():
    # custom dataset folder
    path = '/home/hyerim/data/sets/nuscenes/saved_map/minitrain_intersecion'
    map_list = os.listdir(path)
    idx_list = []
    for i in range(len(map_list)):
        temp = map_list[i].split(".")[0]
        idx = int(temp.split("_")[-1])
        idx_list.append(idx)
    idx_list = sorted(idx_list)
    # print(idx_list)

    with open('wanted_data_idx.pickle','wb') as fw:
        pickle.dump(idx_list,fw)

def has_duplicates2(seq):
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    if not(len(seq) != len(unique_list)):
        return unique_list
    else:
        return seq

def token_save(dataset):
    with open('./wanted_data_idx.pickle','rb') as fw:
        idx_list = pickle.load(fw)

    token_list = []
    for i in range(len(dataset)):
        if i in idx_list:
            instance_token, sample_token = dataset[i].split('_')
            token_list.append(instance_token+"_"+sample_token)
    token_list_unique = has_duplicates2(token_list)

    return token_list_unique


if __name__ == "__main__":
    # intersection_data_save()

    ## for TEST
    train_set = get_prediction_challenge_split("mini_train", dataroot="/home/hyerim/data/sets/nuscenes")
    token_list_unique = token_save(train_set)
    print(token_list_unique)