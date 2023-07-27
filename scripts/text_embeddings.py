import torch
import numpy as np
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import itertools, copy, tqdm


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
      self.model = AutoModel.from_pretrained(model_name)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.seq_length = self.model.max_seq_length
      self.model.to(self.device)
    elif load_method == "Sentence-Transformers":
      from sentence_transformers import SentenceTransformer
      self.model = SentenceTransformer(model_name)
      self.tokenizer = self.model.tokenizer
      self.seq_length = self.model.max_seq_length
      self.model.to(self.device)
    elif load_method == "Instructor":
      from InstructorEmbedding import INSTRUCTOR
      self.model = INSTRUCTOR(model_name)
      self.tokenizer = self.model.tokenizer
      self.seq_length = self.model.max_seq_length
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

    for batch in tqdm.tqdm(batches):
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

  ###This should probably use multiprocessing###
  def split_by_token(self,chunks_w_meta,overlap=True): 
    tokenizer=self.tokenizer
    max_seq_length=self.seq_length
  
    splitter = SentenceSplitter(language='en')
    chunks = [chunk.page_content for chunk in chunks_w_meta]
    new_chunks_w_meta = []

    for i,chunk in enumerate(chunks):
      # Use sentence splitting algorithm
      sentences = splitter.split(text=chunk)
      #Get num of tokens per sentence
      lengths = [len(tokenizer(sentence)['input_ids']) for sentence in sentences]
      new_chunk = []
      #Store total number of tokens for new chunk
      chunk_length = 0
      idn = 0 

      for ii,sentence in enumerate(sentences):
        #Set new chunk length
        chunk_length = chunk_length+lengths[ii]
        #If chunk length more than model sequence length, just add existing chunk to new_chunks
        if chunk_length >= max_seq_length:
          #Fetch metadata for original chunk
          new_meta = copy.deepcopy(chunks_w_meta[i]) #Ensures it doesn't overwrite existing
          new_meta.page_content = ' '.join(new_chunk)
          #Add id to distinguish different parts of the original chunk
          new_meta.metadata['chunk_part'] = idn
          new_meta.metadata['num_tokens'] = chunk_length-lengths[ii]
          new_chunks_w_meta.append(new_meta)
          idn += 1 
          if overlap: 
            new_chunk = [new_chunk[-1],sentence]
            #Make sure chunk_length starts at the length of overlap sentence
            chunk_length = lengths[ii-1]+lengths[ii] 
          else:
            new_chuck = [sentence]
            chunk_length = lengths[ii] 
        else:
          #Keep adding sentences if still within model sequence length
          new_chunk.append(sentence)

      new_meta = copy.deepcopy(chunks_w_meta[i]) #Ensures it doesn't overwrite existing
      new_meta.page_content = ' '.join(new_chunk)
      #Add id to distinguish different parts of the original chunk
      new_meta.metadata['chunk_part'] = idn
      new_chunks_w_meta.append(new_meta)

    return new_chunks_w_meta

