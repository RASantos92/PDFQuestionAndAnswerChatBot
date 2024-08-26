
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from util.singstoreDB import SSDBUtil
load_dotenv()

os.environ["SINGLESTOREDB_URL"] = os.getenv("SINGLESTORE_URL")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("GPT_API_KEY"))

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Gathering bulk PDF information
docsearch = SSDBUtil.gather_documentations("howto-regex.pdf")

print("What do you need help with?")
query = input()

results = docsearch.similarity_search(query=query, k=5)

messages = [("system", system_prompt)]

prompt = ChatPromptTemplate.from_messages(
    messages
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# Invoke the chain with the user's query
results = question_answer_chain.invoke({"input": query, "context": results})
print("Results:", results)  # Print the results if successful

