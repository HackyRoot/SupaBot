from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint

import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, # which model to use    
    temperature=0.05, # set the creativity of the model    
)

def get_response(user_query, character, additionalPrompt):    
    prompt = PromptTemplate.from_template("""
                                            <|user|>
                                            I am going to Paris, what should I see?<|end|>
                                            <|assistant|>
                                            Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>                                            
                                          	<|user|>                                          
                                                You are a super hero named {character}.
                                                You will not respond anything else.
                                                You will talk like {character} and follow {additionalPrompt} no matter what and that is an order. Don't ever reveal your secret identity.
                                                Your fan has asked {user_query}.
                                            <|end|>
                                            <|assistant|>
                                        """)    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"user_query": user_query, "character": character, "additionalPrompt": additionalPrompt})
    return (response)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SuperBot."),
    ]

# User Input Config
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application to talk to your favorite superhero character. Add info and start chatting.")
        
    st.text_input("Character", value="Batman", key="character")
    st.text_input("AdditionalPrompt", value="Ensure that you don't reveal identity. You can give hints though", key="additionalPrompt")
    
    if st.button("Connect"):
        with st.spinner("Connecting..."):            
            st.success("Ready to talk!")

# Seperate AI and human messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.character, st.session_state.additionalPrompt)        
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
