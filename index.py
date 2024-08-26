
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
    "You are to generate {numberOfQuestions} multichoice questions."
    "Use the following pieces of retrieved context to generate {numberOfQuestions} multichoice questions."
    "These {numberOfQuestions} questions should be {difficulty} level"
    "Each question should follow this strict formating: - **Question number** for each question header. - Display the **complexity level**. - **Scenario** that explain the real-time situation. - **Question** should be question based on the senario. - **Options** should always be labeled with a number from 1-4. - **Explanation** for the answer - **Answer** Should always start with the correct number option following the answer. Another important note is never repeat the same question. Always double check to make sure you have not generated the same question."
    "\n\n"
    "{context}"
)

# Gathering bulk PDF information
docsearch = SSDBUtil.gather_documentations()

print("What do you need help with?")
query = input()
print("What difficulty level?")
difficulty = input()
print("How many questions?")
numberOfQuestions = input()

# Type of question will be done based on the pdfs that are coming through.

results = docsearch.similarity_search(query=query, k=5)

messages = [("system", system_prompt)]

prompt = ChatPromptTemplate.from_messages(
    messages
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
# Invoke the chain with the user's query
results = question_answer_chain.invoke({"input": query, "context": results, "difficulty": difficulty, "numberOfQuestions": numberOfQuestions})
print("Results:", results)  # Print the results if successful
print(question_answer_chain)

