from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Added MessagesPlaceholder for history
from langchain_classic.memory import ConversationBufferWindowMemory # Added memory
# OR: from langchain_openai import ChatOpenAI  # paid

# 1. Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load Vector Store
vectordb = Chroma(
    persist_directory="./vectorstore", 
    embedding_function=embeddings
)

# 3. Create Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 4. LLM Setup
llm = ChatOllama(model="llama3.2", temperature=0.1) # Slight temperature helps with "suggestions"
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Replace RetrievalQA with ConversationalRetrievalChain memory logic
memory = ConversationBufferWindowMemory(
    k=5,                            # remember last 5 turns
    memory_key="chat_history",
    return_messages=True
)

# 5. The "Helpful" Prompt
# We changed this to provide suggestions even when exact data is missing.
system_prompt = (
    "You are a helpful assistant. Use the following pieces of retrieved context "
    "to answer the user's question. \n\n"
    "STRICT RULE: If the answer is not explicitly in the context, do not make up facts. "
    "Instead, state that the specific details are missing from the data, but provide "
    "3 helpful suggestions or general guidance related to the topic of the question.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # This is where memory is injected
        ("human", "{input}"),
    ]
)

# 6. Build the Modern Chain (Option 1)
# The "Stuff" chain handles how documents are formatted into the prompt
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# The "Retrieval" chain links the search logic to the LLM logic
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 7. Interactive Loop
print("RAG Chatbot ready. Type 'quit' to exit.\n")
while True:
    query = input("You: ")
    if query.lower() == "quit": 
        break
    
    # Load history for the current turn
    history = memory.load_memory_variables({})["chat_history"]
    
    # In the modern chain, we use 'input' instead of 'query'
    # Query now uses: rag_chain.invoke (equivalent to the requested qa_chain.invoke)
    result = rag_chain.invoke({
        "input": query,
        "chat_history": history
    })
    
    # Save this turn to memory
    memory.save_context({"input": query}, {"output": result['answer']})
    
    print(f"\nBot: {result['answer']}")
    
    print("\nSources used:")
    # Document sources are now in 'context' list
    for doc in result['context']:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"  - {source} (Page: {page})")
    print("-" * 30)