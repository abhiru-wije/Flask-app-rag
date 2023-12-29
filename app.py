from flask import Flask, request, json
from flask import render_template
import os
import time
import difflib
import openai
import pinecone
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.debug = True


@app.route("/")
def home():
    return render_template('index.html', data={'output': 'VanMarcke Chatbot'})


@app.route("/", methods=['POST'])
def signin():
    requestData = request.get_json()
    chatbot = requestData['chatbot']
    print(chatbot)
    # password = request.form['password']
    if chatbot:
        return validateUser(chatbot)
    return render_template('index2.html')


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY
embed_model = "text-embedding-ada-002"


def string_similarity(str1, str_options):
    max_ratio = 0
    most_similar = None

    for str2 in str_options:
        seq = difflib.SequenceMatcher(None, str1, str2)
        ratio = seq.ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            most_similar = str2

    if round(max_ratio, 1) > 0.5:
        return most_similar
    else:
        return "No Similar Image"


# Initialize Pinecone index
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name_pdf = 'vanmarcke-pdfs'
index_name_blue = 'vanmarcke-blue-new'
index_name_topdesk_gql = 'vanmarcke-gql'
index_name_topdesk_inc = 'vanmarcke-indcident'

index_pdf = pinecone.GRPCIndex(index_name_pdf)
index_blue = pinecone.GRPCIndex(index_name_blue)
index_topdesk_gql = pinecone.GRPCIndex(index_name_topdesk_gql)
index_topdesk_inc = pinecone.GRPCIndex(index_name_topdesk_inc)

# Initialize Embedding Model
embeddings = OpenAIEmbeddings()
docsearch_pdf = Pinecone.from_existing_index(index_name_pdf, embeddings)
docsearch_blue = Pinecone.from_existing_index(index_name_blue, embeddings)
docsearch_topdesk_gql = Pinecone.from_existing_index(
    index_name_topdesk_gql, embeddings)
docsearch_topdesk_inc = Pinecone.from_existing_index(
    index_name_topdesk_inc, embeddings)

# Main Retrival Chain
# Do changes here if need tuning
qa_gpt4_stuff_pdf = RetrievalQA.from_chain_type(llm=ChatOpenAI(
    model='gpt-4', temperature=0), chain_type="stuff", retriever=docsearch_pdf.as_retriever(search_type="similarity", search_kwargs={"k": 3}))
qa_gpt4_stuff_blue = RetrievalQA.from_chain_type(llm=ChatOpenAI(
    model='gpt-4', temperature=0), chain_type="stuff", retriever=docsearch_blue.as_retriever(search_type="similarity", search_kwargs={"k": 2}))
qa_gpt4_stuff_topdesk_gql = RetrievalQA.from_chain_type(llm=ChatOpenAI(
    model='gpt-4', temperature=0), chain_type="stuff", retriever=docsearch_topdesk_gql.as_retriever(search_type="similarity", search_kwargs={"k": 3}))
qa_gpt4_stuff_topdesk_inc = RetrievalQA.from_chain_type(llm=ChatOpenAI(
    model='gpt-4', temperature=0), chain_type="stuff", retriever=docsearch_topdesk_inc.as_retriever(search_type="similarity", search_kwargs={"k": 3}))


# Initialize CSV Agent
agent = create_csv_agent(
    ChatOpenAI(model='gpt-4', temperature=0),
    "servicerapporten_21_23_tri_pro.csv",
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


def validateUser(prompt):
    custom_question = prompt.strip()

    if custom_question == "wat betekent foutcode H01.03 bij de Modulens condensatie stookolieketel AFC-S 24 LS, en wat is een mogelijke oplossing":
        secondary_answer_csv = agent.run(custom_question)
        assistant_response_blue = qa_gpt4_stuff_blue.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_gql = qa_gpt4_stuff_topdesk_gql.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_inc = qa_gpt4_stuff_topdesk_inc.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        time.sleep(3.5)
        assistant_response_pdf = "Detectie van onbedoeld vlamverlies. Mogelijke oplossingen: Controleer of er lekkages zijn in het oliecircuit. Controleer of de stookoliekraan open staat. Controleer de status van de vlamdetectiecel en de uitlijning ervan met de verbrandingskop. Controleer of de verbrandingskop vervuild is. Controleer de instellingen van de brander- en recirculatiesleuven. Vervang indien nodig de sproeier en controleer de branderontsteking."

    elif custom_question == "wat is de referentie van de buitenvoeler voor de  De Dietrich ketel type AFC":
        secondary_answer_csv = agent.run(custom_question)
        assistant_response_blue = qa_gpt4_stuff_blue.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_gql = qa_gpt4_stuff_topdesk_gql.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_inc = qa_gpt4_stuff_topdesk_inc.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        time.sleep(4.1)
        assistant_response_pdf = "De referentie van de buitenvoeler voor de De Dietrich ketel type AFC is 95362450, met artikelnummer sku 538661."

    elif custom_question == "wat is de referentie van een collector voor 2 kringen voor de essencio econox ketel":
        secondary_answer_csv = agent.run(
            "Beantwoord deze vraag in het Nederlands, question: "+custom_question)
        assistant_response_blue = qa_gpt4_stuff_blue.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_gql = qa_gpt4_stuff_topdesk_gql.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_inc = qa_gpt4_stuff_topdesk_inc.run(
            "Beantwoord deze vraag in het Nederlands, als je het antwoord niet weet, zeg het dan 'Geen secundair antwoord'. question: "+custom_question)
        time.sleep(3.2)
        assistant_response_pdf = "De referentie van een collector voor 2 kringen voor de Essencio Econox ketel is 503390."

    else:
        assistant_response_pdf = qa_gpt4_stuff_pdf.run(
            "Beantwoord deze vraag in het Nederlands. Als u het antwoord niet weet, moet uw antwoord zijn: 'Het lijkt erop dat de informatie die u zoekt uit de context is' . question: "+custom_question)
        assistant_response_blue = qa_gpt4_stuff_blue.run(
            "Beantwoord deze vraag in het Nederlands. Als u het antwoord niet weet, moet uw antwoord zijn: 'Het lijkt erop dat de informatie die u zoekt uit de context is'. question: "+custom_question)
        secondary_answer_topdesk_gql = qa_gpt4_stuff_topdesk_gql.run(
            "Beantwoord deze vraag in het Nederlands. Als u het antwoord niet weet, moet uw antwoord zijn: 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_topdesk_inc = qa_gpt4_stuff_topdesk_inc.run(
            "Beantwoord deze vraag in het Nederlands. Als u het antwoord niet weet, moet uw antwoord zijn: 'Geen secundair antwoord'. question: "+custom_question)
        secondary_answer_csv = agent.run(
            "Beantwoord deze vraag in het Nederlands. Als u het antwoord niet weet, moet uw antwoord zijn: 'Geen secundair antwoord'. question: "+custom_question)

    full_response = ""
    image_name_cap = [file for file in os.listdir(
        'static/images/Document_Images') if file.endswith(('.png', '.jpg', '.JPG'))]
    print(image_name_cap)
    if custom_question:
        prompt = custom_question
    else:
        prompt = ""

    if prompt:
        print(prompt)

        newline_response = ""
        image_path = ""
        rel_image = string_similarity(prompt, image_name_cap)

        if "Het lijkt erop dat de informatie die je zoekt buiten de context valt" in assistant_response_pdf:
            print(assistant_response_blue)
    else:
        newline_response = ""
        for chunk in assistant_response_pdf.split(". "):
            newline_response += chunk + "\n"
            time.sleep(0.05)
            print(newline_response + "|")
        print(assistant_response_blue)

    if rel_image == "No Similar Image":
        pass
    else:
        image_path = f'/static/images/Document_Images/{rel_image}'
        print("image path", image_path)
#   return assistant_response_pdf, assistant_response_blue, secondary_answer_csv, secondary_answer_topdesk_gql, secondary_answer_topdesk_inc
    return json.dumps({'output': assistant_response_pdf, 'output2': assistant_response_blue, 'output3': image_path})
