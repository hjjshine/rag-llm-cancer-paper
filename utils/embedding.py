# utils/embedding.py
import faiss
import numpy as np

# Get text embeddings
def get_text_embedding(input, CLIENT, model_embed):
    if model_embed == 'mistral-embed':
        embeddings_batch_response = CLIENT.embeddings(
            model=model_embed,
            input=input
        )    
    else:
        embeddings_batch_response = CLIENT.embeddings.create(
            model=model_embed,
            input=input
        )
    return embeddings_batch_response.data[0].embedding

# Store context embeddings in a vector database
def store_embedding(context_embeddings):
    context_embeddings = context_embeddings.astype('float32') 
    d = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(context_embeddings)
    return(index)

# Prepare embedding for index search
def prep_embed_for_search(embedding, n_dim):
    embedding = np.array(embedding, ndmin=n_dim)
    embedding = embedding.astype('float32') 
    return(embedding)

# Generate context vector database
def get_context_db(context_chunks, CLIENT, model_embed):
    context_embeddings=np.array([get_text_embedding(chunk, CLIENT, model_embed) for chunk in context_chunks])
    index=store_embedding(context_embeddings)
    return(index)

# Retrieve top k context chunks based on similarity search
def retrieve_context(context_chunks, prompt_chunk, CLIENT, model_embed, index, num_vec):
    query_embeddings=np.array([get_text_embedding(prompt_chunk, CLIENT, model_embed)])
    D, I = index.search(prep_embed_for_search(query_embeddings, n_dim=2), k=num_vec)
    retrieved_chunk = [context_chunks[i] for i in I.tolist()[0]]
    return(retrieved_chunk)