import torch
import numpy as np
from sklearn.metrics import pairwise
class Retriever():
  def __init__(self,ReRanker=None,EmbeddingModel=None, MetaStore=None, Faiss_Index=None,batch_size=10):
    self.ReRanker = ReRanker
    self.EmbeddingModel = EmbeddingModel
    self.MetaStore = MetaStore
    self.Faiss_Index = Faiss_Index

  def index_retrieve(self, query, num_results=10, threshold=0.2, fields=[]):
    if fields==[]: fields=list(self.MetaStore.df.columns)
    query = self.EmbeddingModel.embed_text([query])
    distances, indexes = self.Faiss_Index.search(query=query,k=num_results)
    data={'distances':[],'indexes':[]}
    for d,i in zip(distances[0],indexes[0]):
      if d>0.3:
        data['distances'].append(d)
        data['indexes'].append(i)
    #Order from high to low
    data['distances'].reverse()
    data['indexes'].reverse()
    #Get Metadata
    metadata = self.MetaStore.get_rows_by_chunk_id(data['indexes'])
    return [{'distance':d,'index':i,'metadata':m[1][fields],'collection_name':self.MetaStore.collection_name} for d,i,m in zip(data['distances'], data['indexes'], metadata.iterrows())]

  def simply_score(passages,query,num_results=5,threshold=0.2, rerank=False):
    #Embed Query and Passages
    passage_embeddings = self.EmbeddingModel.embed_text(passages)
    query_embedding = self.EmbeddingModel.embed_text([query])[0]

    #Calculate simple cosine similarity
    similarities = pairwise.cosine_similarity(query_embedding,passage_embeddings)[0]
    sim_scores_argsort = reversed(np.argsort(similarities))

    #Store results>threshold in dict
    results = [{'index':i,'score':similarities[i]} for i in sim_scores_argsort if similarities[i] > threshold]
    
    if rerank:
        passages_for_rank = [i for i in results['index']]
        ranked = self.ReRanker.rerank(passages_for_rank,query,n_results,threshold)
        ranked['original_score'] = [similarities[ii] for ii in [ranked[i]['index'] for i,n in enumerate(ranked)]]
        results = ranked
      
    return results


class ReRanker():
  def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda", batch_size=25):
    #Set device
    if not device == "cpu":
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = 'cpu'

    #Load reranker
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name, device=device) 
    self.model = model
    self.batch_size = batch_size

  def rerank(self, passages,query,n_results,threshold=0.3):
    batches =  [passages[i : i + self.batch_size] for i in range(0, len(text), self.batch_size)]
    scores=[]
    for batch in batches:
      model_inputs = [[query, passage] for passage in batch]
      b_scores = self.model.predict(model_inputs)
      scores.append(b_scores)

    #Sort the scores in decreasing order
    sim_scores_argsort = reversed(np.argsort(b_scores))

    #Limit to results above threshold
    results = [{'index':i,'ranked_score':similarities[i]} for i in sim_scores_argsort if similarities[i]>threshold]
  
    return results
