import streamlit as st

import os

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks import get_openai_callback
import openai
st.set_page_config(page_title= "The Joe RAGan Experience", page_icon="🤣")



# st.session_state.user = st.sidebar.text_input("Enter your name")

# Lets implement the enter api key function
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai.api_key = openai_api_key
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


# Connect to and create weaviate vectorstores:
import weaviate
auth_config = weaviate.AuthApiKey(api_key="J12sontXiJhQuABUlZoXL4lgTDmhjsF4dO05")

client = weaviate.Client(
    url="https://streamlit-llm-hackathon-vb29y6cc.weaviate.network",
    auth_client_secret=auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": openai_api_key
    }
)

from langchain.vectorstores import Weaviate
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key= openai_api_key)
chunk_vectorstore = Weaviate(client, index_name = "Chunk_Node", text_key = "chunk_body", attributes=['ep_title','ep_link'])
topic_vectorstore = Weaviate(client, index_name = "Topic_Node", text_key = "topic")


#Custom Search Function using Topics Filter:
def search_with_topics(question):
    topics = topic_vectorstore.similarity_search(question)
    topicList = []
    for doc in topics:
        topicList.append(doc.page_content)
    print("List of relevant topics: {}".format(topicList))

    #construct where filter with relevant topic list
    where_filter = {
            "path": ["chunk_topics"],
            "operator": "ContainsAny",
            "valueText": topicList     
    }

    #perform search of chunks with relevant topics
    chunks = chunk_vectorstore.similarity_search(question, where_filter = where_filter, additional = ["certainty"])
    return chunks



# Create the sessionstate tools and agent:
from langchain.tools import Tool
if 'tools' not in st.session_state:
   st.session_state.tools = [
    Tool.from_function(
    name = "joerogansearch",
    description="Searches and returns documents from the Joe Rogan knowledge base to answer any questions regarding content from the podcast episode.",
    func=search_with_topics,)
]

from langchain.schema import SystemMessage
sys_mes_joe = SystemMessage(content="Do your best to answer the questions. The questions will all be about the content of epsiodes from the Joe Rogan Podcast. Use the joerogansearch tool to look up relevant information. When using the context from the tool be sure to cite the source by providing the URL Link at the end of your response.")


from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
if 'joe_rogan_agent' not in st.session_state:
   llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", streaming=True)
   st.session_state.joe_rogan_agent = create_conversational_retrieval_agent(llm, st.session_state.tools, verbose = False, system_message=sys_mes_joe)
   


# Set up the streamlit interface (rough draft):

avatars = {"user": '🧠', "assistant": '🤖'}

if "messages" not in st.session_state:
   st.session_state["messages"] = [{"role":"assistant", "content": "What would you like to know?"}]

if "tokens" not in st.session_state:
    st.session_state.tokens = ["Tokens Used: 0 Prompt Tokens: 0 Completion Tokens: 0 Successful Requests: 0 Total Cost (USD): $0.0"]

if st.sidebar.button("Clear message history"):
    st.session_state.messages = [{"role":"assistant", "content": "What would you like to know?"}]
    llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", streaming=True)
    st.session_state.joe_rogan_agent = create_conversational_retrieval_agent(llm, st.session_state.tools, verbose = True, system_message=sys_mes_joe)

for n,msg in enumerate(st.session_state.messages):
   st.chat_message(msg["role"],avatar=avatars[msg["role"]]).write(msg["content"])

user_query = st.chat_input(placeholder="Enter Question")

if user_query:
   st.session_state.messages.append({"role": "user", "content": user_query})
   st.chat_message("user", avatar=avatars["user"]).write(user_query)

   with st.chat_message("assistant", avatar=avatars["assistant"]):
    #We can use the streamlit call back handler to stream responses but it is mad tricky. Would be nice to find a cleaner solution or implementation.
    st_cb = StreamlitCallbackHandler(st.container())
    with get_openai_callback() as cb:
        response = st.session_state.joe_rogan_agent({"input": user_query}, callbacks = [st_cb,cb])
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.session_state.tokens.append(cb)
        st.write(response["output"])
        st.write(st.session_state.tokens[-1])
