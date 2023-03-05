from transformers import BertModel, BertTokenizer

bert = "bert-base-cased"
bert_path = "./bert-base-cased/"
tokenizer = BertTokenizer.from_pretrained(bert)
model = BertModel.from_pretrained(bert)

tokenizer.save_pretrained(bert_path)
model.save_pretrained(bert_path)
