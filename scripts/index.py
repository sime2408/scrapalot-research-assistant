import faiss
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq

class Faiss_Index:
    def __init__(self, shape=None, db_name="my_documents", db_dir='db'):
        index_dir =  os.path.join(db_dir, db_name)
        index_path = os.path.join(index_dir, "/index.index")
        # Check if index already exists
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            os.makedirs(index_dir,exist_ok=True)
            index = faiss.IndexFlatL2(384)
            faiss.write_index(index, index_path)
        self.index_path = index_path
        self.index = index
    
    def search(self, query, k):
        return self.index.search(query, k)

    def load(self,index_path):
        # Load index from disk
        self.index = faiss.read_index(index_path)

    def add(self, embeddings, chunk_ids):
        self.index.add(embeddings)
        self.index.save()

        #Get indexes of recently added chunks
        start_index = index.ntotal - num_added
        index_list = list(range(start_index, index.ntotal))
        MetaStore.mark_indexed(chunk_ids,index_list)

    def reset(self):
        # Reset and reload index
        self.index.reset()
        self.load()

    def get_embeddings(self, indexes):
        return self.index.reconstruct_n(indexes)


class MetaStore:
    def __init__(self, db_name="my_documents", db_dir="metadata", columns = ['file_path', 'file_id', 'chunk_id', 'chunk_text', 'is_indexed', 'index_id', 'custom_metadata']):
        index_dir =  os.path.join(db_dir, db_name)
        parquet_path = os.path.join(index_dir, 'metadata.parquet')
        
        if os.path.exists(parquet_path):
            # Load existing Parquet file
            df = pd.read_parquet(parquet_path)
        else:
            # Create new Parquet file
            os.makedirs(index_dir,exist_ok=True)
            df = pd.DataFrame(columns=columns)
            df.to_parquet(parquet_path)

        self.parquet_path = parquet_path
        self.df = df

    def add_chunk(self, chunk_dicts):
        self.df = self.df.append(chunk_dicts, ignore_index=True)
        self.save()
        
    def mark_indexed(self, chunk_ids, index_ids):
        self.df.loc[chunk_ids, 'is_indexed'] = True
        self.df.loc[chunk_ids, 'index_id'] = index_ids
        self.save()

    def get_rows_by_chunk_id(self, chunk_ids):
        """Fetch rows by list of chunk ids"""
        return self.df.loc[self.df['chunk_id'].isin(chunk_ids)]

    def save(self):
        # Save DataFrame back to Parquet
        self.df.to_parquet(self.parquet_path)
