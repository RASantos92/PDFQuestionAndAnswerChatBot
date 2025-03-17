import os

from dotenv import load_dotenv
from util.singstoreDB import SSDBUtil

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

# Setting up my connection to singlestore DB
os.environ["SINGLESTOREDB_URL"] = os.getenv("SINGLESTORE_URL")

# Obtaining LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("GPT_API_KEY"))

system_prompt = (
    "You are to generate {numberOfQuestions} multichoice questions."
    "Use the following pieces of retrieved context to generate {numberOfQuestions} multichoice questions."
    "These {numberOfQuestions} questions should be {difficulty} level"
    "Each question should follow this strict formating: - **Question number** for each question header. - Display the **complexity level**. - **Scenario** that explain the real-time situation. - **Question** should be question based on the senario. - **Options** should always be labeled with a number from 1-4. - **Explanation** for the answer - **Answer** Should always start with the correct number option following the answer. Another important note is never repeat the same question. Always double check to make sure you have not generated the same question."
    "\n\n"
    "{context}"
)

# Gathering bulk PDF information.
#   This can be modularized to only grab certain 
#   chunks of data based on the subject.
docsearch = SSDBUtil.gather_documentations()

# Used to query the gathered documents.
#   This can be used to query sub data needed from the db
#   instead of gathering the whole subject.
print("What do you need help with?")
query = input()

# Information gathered by the user will be used to populate the system_prompt variable.
print("What difficulty level?")
difficulty = input()
print("How many questions?")
numberOfQuestions = input()

# Type of question will be done based on the pdfs that are coming through.
similarity_results = docsearch.similarity_search(query=query, k=5)

# Prompting the LLM with the system_prompt
messages = [("system", system_prompt)]
prompt = ChatPromptTemplate.from_messages(
    messages
)

# Here we are initiating a chain, specifially a stuff chain.
#   "Stuff chain" is a convinient way of combining multiple documents (or text chunks) in one prompt.
#   Promt chaining, each LLM call processes the output of the previous one. 
#   To be more token efficient use (map-reduce, refine), also can add checks (gates).
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Invoke the chain with the user's query, and the similarity results as context
results = question_answer_chain.invoke({"input": query, "context": similarity_results, "difficulty": difficulty, "numberOfQuestions": numberOfQuestions})
print("-"*100,"\n", similarity_results,"-***"*100,"\n")
print("Results:", results)  # Print the results if successful
print(question_answer_chain)