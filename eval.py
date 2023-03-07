import torch
import sys

sys.path.append(".")
import torch.optim as optim
from models.bert_em import BERT_EM
from models.proto_sim import ProtoSimModel
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer
from dataset.dataset import Data
from transformers import AdamW, WarmupLinearSchedule
import os
import json

from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("device", device)

dim_reducer = TSNE(n_components=2)


def visualize_embeddings(hidden_states, labels, title):

    fig = plt.figure(figsize=(24, 24))

    labels = np.array(labels)

    layer_dim_reduced_embeds = dim_reducer.fit_transform(hidden_states.numpy())

    df = pd.DataFrame.from_dict(
        {
            "x": layer_dim_reduced_embeds[:, 0],
            "y": layer_dim_reduced_embeds[:, 1],
            "label": labels,
        }
    )
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="Set2", s=400)
    plt.savefig(f"plots/{title}", format="png", pad_inches=0)


def flat(batch):
    # batch, sample_number, data -> batch * sample_number, data
    size = batch.size()
    if len(size) == 3:
        return batch.view(-1, size[-1])
    elif len(size) == 2:
        return batch.view(size[0] * size[1])
    else:
        return batch


def load_model(ckpt):

    return BERT_EM.from_pretrained(ckpt)


if __name__ == "__main__":

    with torch.no_grad():
        model_dir = "ckpts/ckpt_599/"
        model_dir = "bert-base-cased"
        bert_encoder = BERT_EM.from_pretrained(model_dir)
        bert_encoder.to(device)
        print("Finished loading pre-trained model...")

        config = json.load(open("./control/train_config.json", "r"))

        eval_data = Data(config["eval_data"], config)
        labels2ids = eval_data.labels2ids
        ids2labels = {v: k for k, v in labels2ids.items()}

        print(labels2ids)

        # bert_path = "/home/users/atuo/language_models/bert/bert-base-cased/"
        bert_path = "bert-base-cased"
        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)

        # proto_sim_model = ProtoSimModel(config["relations"], config["embedding_size"])

        # proto_sim_model.to(device)

        labels = []
        preds = []

        # choose random batch
        i_batch = torch.randint(0, len(eval_data), (1,)).item()

        batch_data = eval_data[i_batch]

        p_input_ids = batch_data["p_input_id"]
        p_mask = batch_data["p_mask"]
        p_e_pos1 = batch_data["p_e_pos1"]
        p_e_pos2 = batch_data["p_e_pos2"]
        p_labels = batch_data["p_label"]
        print("p_labels: ", p_labels)

        # support set
        p_relation_embedding = bert_encoder.predict(
            p_input_ids.to(device),
            p_e_pos1.to(device),
            p_e_pos2.to(device),
            attention_mask=p_mask.to(device),
        )

        n_input_ids = batch_data["n_input_id"]
        n_mask = batch_data["n_mask"]
        n_e_pos1 = batch_data["n_e_pos1"]
        n_e_pos2 = batch_data["n_e_pos2"]
        n_labels = batch_data["n_label"]

        n_relation_embedding = bert_encoder.predict(
            n_input_ids.to(device),
            n_e_pos1.to(device),
            n_e_pos2.to(device),
            attention_mask=n_mask.to(device),
        )
        print("n_relation_embedding: ", n_relation_embedding.shape)
        print("p_relation_embedding: ", p_relation_embedding.shape)

        labels = torch.cat([p_labels, n_labels], dim=0)
        print("labels: ", labels)
        labels = [ids2labels[label] for label in labels.tolist()]
        embeddings = torch.cat([p_relation_embedding, n_relation_embedding], dim=0)

        visualize_embeddings(embeddings, labels, "embeddings.png")

    """
    # prototypes (mean of support set)
    p_prototypes = torch.mean(p_relation_embedding, dim=0).view(1, -1)
    print("p_prototypes: ", p_prototypes.shape)

    n_prototypes = torch.mean(
        n_relation_embedding.view(config["N"] - 1, config["K"], -1), dim=1
    )
    print("n_prototypes: ", n_prototypes.shape)

    # cat prototypes
    prototypes = torch.cat([p_prototypes, n_prototypes], dim=0)
    print("prototypes: ", prototypes.shape)

    # query sets
    config["K"] = 1
    eval_data = Data(config["eval_data"], config)

    total = 0
    for i_batch, batch_data in enumerate(eval_data):

        p_input_ids = batch_data["p_input_id"]
        p_mask = batch_data["p_mask"]
        p_e_pos1 = batch_data["p_e_pos1"]
        p_e_pos2 = batch_data["p_e_pos2"]
        p_labels = batch_data["p_label"]

        p_relation_embedding = bert_encoder.predict(
            p_input_ids.to(device),
            p_e_pos1.to(device),
            p_e_pos2.to(device),
            attention_mask=p_mask.to(device),
        )
        print("p_relation_embedding: ", p_relation_embedding.shape)

        similarity = 1 - 1 / (
            1
            + torch.exp((torch.sum(prototypes * p_relation_embedding, 1) - 1536) / 100)
        )
        print("similarity: ", similarity.shape)
        pred = torch.argmax(similarity, dim=-1)
        pred = pred.cpu().tolist()
        p_labels = p_labels.cpu().tolist()

        print("label: ", p_labels)
        print("pred: ", pred)
        print("-" * 20)

        labels.append(0)
        preds.append(pred)

    score = accuracy_score(y_true=labels, y_pred=preds)
    print("Accuracy: ", score)
    """
