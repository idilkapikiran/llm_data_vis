import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoConfig, AutoModel, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from pathlib import Path


def save_model_locally(model, model_path="local_model_3B"):
    model.save_pretrained(model_path)
    #tokenizer.save_pretrained("./tokenizer")

@st.cache_resource()
def load_and_save_model():
    model_dir = Path("local_model_3B")
    
    #if model_dir.exists():
    #    model = AutoModel.from_pretrained("./local_model_3B")
    #    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    #    pipe = pipeline("text-generation", model=model, device_map="auto")
    #else:'''
    #model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B", device_map="auto")
    save_model_locally(pipe.model, model_path="local_model_3B")
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    return hf_pipeline

def analyze_csv(df):
    analysis = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "numerical_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return analysis

def recommend_visualizations(analysis):
    recommendations = []
    
    if analysis["categorical_columns"]:
        recommendations.append("Bar chart (for categorical data)")
        recommendations.append("Pie chart (for categorical data)")
    
    if analysis["numerical_columns"]:
        recommendations.append("Histogram (for numerical data)")
        recommendations.append("Box plot (for numerical data)")
    
    if analysis["datetime_columns"]:
        recommendations.append("Line chart (for time-series data)")
    
    if len(analysis["numerical_columns"]) >= 2:
        recommendations.append("Scatter plot (for relationships between two numerical columns)")
    
    return recommendations

def main():
    st.title("Visualization Recommender")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data:")
        st.write(df.head())
        
        analysis = analyze_csv(df)
        st.write("### CSV Analysis:")
        st.write(f"Number of Rows: {analysis['num_rows']}")
        st.write(f"Number of Columns: {analysis['num_columns']}")
        st.write(f"Columns: {analysis['columns']}")
        st.write(f"Data Types: {analysis['data_types']}")
        st.write(f"Numerical Columns: {analysis['numerical_columns']}")
        st.write(f"Categorical Columns: {analysis['categorical_columns']}")
        st.write(f"Datetime Columns: {analysis['datetime_columns']}")
        st.write(f"Missing Values: {analysis['missing_values']}")
        
        # Recommend visualizations
        recommendations = recommend_visualizations(analysis)
        st.write("### Recommended Visualizations:")
        st.write(recommendations)
        
        llm = load_and_save_model()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(analysis, embeddings)
        retriever = vectorstore.as_retriever()
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, retriever = retriever)
        qa_system_prompt = (
                "You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above"
        )
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        user_input = st.chat_input("Ask me anything about the CSV!")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            full_input = f"{qa_system_prompt}\n\nUser Question: {user_input}"
            
            response = qa_chain.invoke(full_input)
            print('RESPONSE', response)
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()