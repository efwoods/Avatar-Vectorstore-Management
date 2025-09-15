At this endpoint (@router.post("/{collection_name}/query") I need to pass a query to the chroma_db vectorstore and receive a response. I will be using the response for model context:

This is an example of using the response from the chroma_db vectorstore for model context. In production I will have two docker containers where a fast api will accept a query string, pass that query to this docker container at this query endpoint and use the response in the prompt context in a similar fashion as below:

def generate_with_context(user_input: str, top_k: int = 10, max_new_tokens: int = 50) -> str:
    # Embed query and retrieve from vector DB
    query_vec = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    print(f"results: {results}")

    docs = results.get("documents", [[]])[0]
    if not docs:
        context = "No relevant context found."
    else:
        # Optionally truncate context length for model input token limits
        # Here we join and limit length (e.g., first 1000 chars)
        context = "\n".join(docs)
        context = context[:1000]

    # Improved prompt with explicit instruction and clear delimiters
    prompt = f"""You are the person in the contextual statements. Use the context to answer the question briefly and only once.

Context:
{context}

Q: {user_input}
A:"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Set True to enable sampling
            # temperature=0.7,  # Uncomment if do_sample=True
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after last "A:" in case prompt or output has multiple
    answer = decoded.split("A:")[-1].strip()
    return answer
generate_with_context(user_input=input())

# This is the actual endpoint to be updated:


@router.post("/{collection_name}/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query collection",
    description="Query documents in a collection using similarity search"
)
async def query_collection(
    collection_name: str,
    query_request: QueryCollectionRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Query documents in a collection"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        query_request.collection_name = processed_name
        
        result = await chroma_manager.query_collection(processed_name, query_request.query)
        return result
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to query collection: {str(e)}"
        )
