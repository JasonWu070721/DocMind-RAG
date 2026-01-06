from core.llm.local_llm import get_llm
from core.vectorstore.chroma_store import get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

md_text = r"D:/github/DocMind-RAG/core/ocr/output/print-tech-outlier-01-001_res.json"

jq_schema = (
    '.parsing_res_list[] '
    '| select(.block_label == "text" or .block_label == "table")'
)

loader = JSONLoader(
    file_path=md_text,
    jq_schema=jq_schema,
    text_content=False,
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

vector_store = get_vectorstore()
print(vector_store)
vector_store.add_documents(chunks)
vector_store.persist()

exit()

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# llm = get_llm()
# ai_msg = llm.invoke(messages)
# print(ai_msg)
