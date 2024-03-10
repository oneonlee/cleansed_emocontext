from helpers import normalize
import pandas as pd
from datasets import load_dataset

label_dict = {
    0: "others",
    1: "happy",
    2: "sad",
    3: "angry",
}

dataset = load_dataset("emo")

train_df = pd.read_csv("../data/train.txt", delimiter='\t')
dev_df = pd.read_csv("../data/dev.txt", delimiter='\t')
test_df = pd.read_csv("../data/testwithoutlabels.txt", delimiter='\t')
test_df["label"] = [label_dict[label_idx] for label_idx in dataset["test"]["label"]]

if __name__ == "__main__":

    # train
    train_df = train_df.drop(["id"], axis=1)
    for col in ["turn1", "turn2", "turn3"]:
        train_df[col] = train_df[col].apply(normalize)
    train_df.to_csv("../data/cleansed_train.csv", index=False)
    
    # dev
    dev_df = dev_df.drop(["id"], axis=1)
    for col in ["turn1", "turn2", "turn3"]:
        dev_df[col] = dev_df[col].apply(normalize)
    dev_df.to_csv("../data/cleansed_dev.csv", index=False)

    # test
    test_df = test_df.drop(["id"], axis=1)
    for col in ["turn1", "turn2", "turn3"]:
        test_df[col] = test_df[col].apply(normalize)
    test_df.to_csv("../data/cleansed_test.csv", index=False)