import streamlit as st 
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

DB_FAISS_PATH = 'vectorstore/db_faiss'
CSV_PATH = 'RBI DATA states_wise_population_Income.csv'  # Specify the path to your CSV file here

# Function to load the LLM model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Streamlit app title and header
st.title("GRUS AND GRADE BOT🧑‍🌾🌱")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://www.grusandgrade.com/contact-us/index.html'>Our Website </a></h3>", unsafe_allow_html=True)

# Main app logic
# Loading CSV data
loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Embeddings and vector store initialization
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)

# Load LLM model
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

# Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about the our website 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
# Container for chat history
response_container = st.container()

# Container for user's text input
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    # Handle user input
    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)



# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
