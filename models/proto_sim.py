import torch
from torch import nn


class ProtoSimModel(nn.Module):
    def __init__(self, relation_count, embedding_width):
        nn.Module.__init__(self)
        self.prototypes = nn.Embedding(relation_count, embedding_width)
        self.classification_layer = nn.Linear(embedding_width, relation_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, relation_embedding, relation_id):
        protos = self.prototypes(relation_id)

        similarity = 1 - 1 / (
            1 + torch.exp((torch.sum(protos * relation_embedding, 1) - 384) / 100)
        )
        predict_relation = self.classification_layer(protos)
        return similarity.cuda(), predict_relation.cuda()

    def predict(self, relation_embedding, relation_id):
        protos = self.prototypes(relation_id)
        similarity = 1 - 1 / (
            1 + torch.exp((torch.sum(protos * relation_embedding, 1) - 384) / 100)
        )  #
        predict_relation = self.classification_layer(protos)
        return similarity.to(self.device), predict_relation.to(self.device)
