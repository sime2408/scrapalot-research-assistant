import torch
import numpy as np

class EmbeddingModel():
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",load_method='Sentence-Transformers', device="cuda", batch_size=50):
    #Set device
    if not device == "cpu":
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = 'cpu'

    #Load text embeddings model
    if load_method == 'AutoModel':
      from transformers import AutoTokenizer, AutoModel
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModel.from_pretrained(model_name)
      self.model.to(self.device)
    elif load_method == "Sentence-Transformers":
      from sentence_transformers import SentenceTransformer
      self.model = SentenceTransformer(model_name)
      self.model.to(self.device)
    elif load_method == "Instructor":
      from InstructorEmbedding import INSTRUCTOR
      self.model = INSTRUCTOR(model_name)
      self.model.to(self.device)
    else:
      System.exit("Invalid load method")

    self.load_method = load_method
    self.batch_size = batch_size

  def mean_pooling(self, token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

  def getTokenLength(self, passages):
    batches =  [passages[i : i + self.batch_size] for i in range(0, len(passages), self.batch_size)]
    token_lengths = []
    for batch in batches:
      token_lengths.append(self.model.tokenizer(passages))
    return token_lengths

  def embed_text(self, passages, instruction=None):
    batches =  [passages[i : i + self.batch_size] for i in range(0, len(passages), self.batch_size)]
    all_embeddings = []
    for batch in batches:
      if instruction!=None:
        batch=[[instruction,sentence] for sentence in batch]

      if self.load_method == "AutoModel":
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        embeddings = np.asarray(embeddings.cpu().detach().numpy())
      else:
        embeddings = self.model.encode(batch)

      all_embeddings.extend(embeddings)
    return all_embeddings
