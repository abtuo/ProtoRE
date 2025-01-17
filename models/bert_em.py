import torch
import sys

sys.path.append("..")
from transformers.modeling_bert import *


class BERT_EM(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_EM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        e_pos1,
        e_pos2,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_input_ids=None,
        masked_lm_labels=None,
        dropout=torch.nn.Dropout(0),
    ):
        try:
            bert_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )
        except:
            print("input_ids: ", input_ids)
            print("e_pos1: ", e_pos1)
            print("e_pos2: ", e_pos2)
            print("attention_mask: ", attention_mask.shape)
            raise Exception("Error in BERT_EM forward")

        # get relation representation
        sequence_output = bert_output[0]  # batch_size * sequence_length * hidden_size
        factor = torch.tensor(range(0, e_pos1.shape[0])).cuda()
        unit = input_ids.shape[1]
        offset = factor * unit
        e_pos1 = e_pos1 + offset
        e_pos2 = e_pos2 + offset

        start_embedding_e1 = torch.index_select(
            sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos1
        )  # batch_size * hidden_size
        start_embedding_e2 = torch.index_select(
            sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos2
        )
        relation_embedding = torch.cat((start_embedding_e1, start_embedding_e2), 1)
        if torch.cuda.is_available():
            start_embedding_e1 = start_embedding_e1.cuda()
            start_embedding_e2 = start_embedding_e2.cuda()
            relation_embedding = relation_embedding.cuda()

        # trigger_loss = self.trigger_sim(start_embedding_e1, start_embedding_e1)

        trigger_loss = torch.tensor([0.0], requires_grad=True).cuda()

        return relation_embedding, trigger_loss

    def predict(
        self,
        input_ids,
        e_pos1,
        e_pos2,
        attention_mask=None,
        dropout=torch.nn.Dropout(0),
    ):
        try:
            bert_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
            )
        except:
            print("input_ids: ", input_ids)
            print("e_pos1: ", e_pos1)
            print("e_pos2: ", e_pos2)
            print("attention_mask: ", attention_mask)
            raise Exception("Error in BERT_EM forward")

        # get relation representation
        sequence_output = bert_output[0]  # batch_size * sequence_length * hidden_size
        factor = torch.tensor(range(0, e_pos1.shape[0])).to(self.device)
        unit = input_ids.shape[1]
        offset = factor * unit
        e_pos1 = e_pos1 + offset
        e_pos2 = e_pos2 + offset

        start_embedding_e1 = torch.index_select(
            sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos1
        )  # batch_size * hidden_size
        start_embedding_e2 = torch.index_select(
            sequence_output.view(-1, sequence_output.shape[2]), 0, e_pos2
        )
        relation_embedding = torch.cat((start_embedding_e1, start_embedding_e2), 1)

        relation_embedding = relation_embedding.to(self.device)

        return relation_embedding

    def trigger_sim(self, a, b, eps=1e-8):

        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        return torch.mm(a_norm, b_norm.transpose(0, 1)).mean()
