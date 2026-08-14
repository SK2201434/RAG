from app.embeddings.embedder import create_embedding_model

embedding_model = create_embedding_model()

text = "Employees are eligible for 26 weeks of parental leave."

vector = embedding_model.embed_query(text)

print("Vector length:", len(vector))

print("First 10 values:")

print(vector[:10])