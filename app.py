import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from pathlib import Path
import torch
from grounding_sam import segment_and_save_views, caption_object
import tempfile
torch.classes.__path__ = []

def save_model_locally(model, tokenizer, model_path, tokenizer_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

def refine_input(user_input, object_caption, llm):
    # System prompt for the refinement task
    refinement_prompt = f"""
    You are a 3D object segmentation assistant. Your task is to:
    1. Analyze the user's requested object part against the actual object description
    2. Extract the core component/part name (e.g., "ear" from "bunny's ear")
    3. Validate if the part makes sense for this object
    
    Object Description: {object_caption}
    User Request: {user_input}
    
    Examples:
    1. If object is "a cat" and request is "bunny's ear":
    <Part>ear</Part>
    <Validation>This appears to be a cat rather than a bunny. I'll segment the ear.</Validation>
    
    2. If object is "a chair" and request is "the back support":
    <Part>back support</Part>
    <Validation>None</Validation>
    
    3. If object is "a car" and request is "the ears":
    <Part>None</Part>
    <Validation>Cars don't have ears. Did you mean mirrors, antennas, or another part?</Validation>
    """
    
    # Get LLM response
    response = llm(refinement_prompt)
    
    # Parse the response
    try:
        part = response.split("<Part>")[1].split("</Part>")[0].strip().lower()
        validation = response.split("<Validation>")[1].split("</Validation>")[0].strip()
        if validation.lower() == "none":
            validation = None
    except:
        part = user_input
        validation = "Could not parse response - using original input"
    
    # Handle cases where no valid part was extracted
    if part.lower() == "none":
        return None, validation
    
    return part

@st.cache_resource()
def load_and_save_model():
    model_path = Path("./local_model_3B")
    tokenizer_path = Path("./local_model_3B/tokenizer")

    #if model_path.exists() and tokenizer_path.exists():
    #    print('here')
    #    model = AutoModel.from_pretrained(str(model_path))
    #    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    
    #else:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    #save_model_locally(model, tokenizer, model_path=model_path, tokenizer_path=tokenizer_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    return hf_pipeline

def chat():
    torch.set_num_threads(1)
    st.title("3D Part Segmentation Visualization with 2D Multiviews")
    llm = load_and_save_model()
    uploaded_file = st.file_uploader("Upload a 3D object file (i.e ply, obj)", type=["ply", "obj"])

    if uploaded_file:
        save_path = os.path.join('./objects', uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Saved to {save_path}")

        object_caption, mesh_name = caption_object(save_path)
        
        if object_caption:
            st.write("### Object Caption:")
            st.write(object_caption)
        while not object_caption: # test it out
            st.error("Failed to generate object caption. Please check the file format and try again.")
            uploaded_file = st.file_uploader("Upload a 3D object file (i.e ply, obj)", type=["ply", "obj"])

    # Initialize messages if not in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages from the chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input for segmentation
    user_input = st.chat_input("Which part of the object do you want to segment?")
    if user_input:
        # Append user input to the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Segment the object based on the refined input
        #refined_input = refine_input(user_input, object_caption, llm)
        print('REFINED INPUT', user_input)
        
        segmentations = segment_and_save_views(user_input, mesh_name)

        if segmentations:
            for i, segmentation in enumerate(segmentations):
                st.image(segmentation, caption=f"Segmented View {i+1}", use_column_width=True)
            # Set up the embeddings and retriever for the conversation
            #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            #vectorstore = FAISS.from_texts(segmentations, embeddings)
            #retriever = vectorstore.as_retriever()

            # Initialize memory and QA chain
            #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            #qa_chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, retriever=retriever)

            # Define the system prompt
            qa_system_prompt = (
                "You are a helpful assistant that helps the user to segment 3D objects using 2D multiviews. "
                "Based on the user prompt, find the part that they want to segment. Consider the GroundingDINO model is used "
                "to segment the object, therefore simplify which part they want to segment."
            )

            # Full input for the QA system
            full_input = f"{qa_system_prompt}\n\nUser Question: {user_input}"

            # Get the response from the QA chain
            #response = qa_chain.invoke(full_input)
            #print('RESPONSE', response)

            #if 'answer' in response:
            #    # Append assistant's response to chat history and display it
            #    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            #    st.chat_message("assistant").write(response['answer'])
            #else:
            #    st.error("No answer received from the system.")
        else:
            st.error("No segmentations found. Please check the file format and try again.")

           
        
        

if __name__ == "__main__":
    chat()