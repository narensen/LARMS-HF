import streamlit as st
import torch
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import time

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_inference_client(token):
    """
    Initialize Hugging Face Inference Client with error handling
    """
    try:
        client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=token
        )
        return client
    except Exception as e:
        st.error(f"Client Initialization Error: {e}")
        st.error("Please verify your Hugging Face token and network connection.")
        return None

def load_or_compute_embeddings(df, model):
    """
    Load or compute embeddings with timestamp-based refresh
    """
    embeddings_file = 'corpus/embeddings.pt'
    
    # Add timestamp-based recomputation
    if os.path.exists(embeddings_file):
        try:
            file_age = time.time() - os.path.getmtime(embeddings_file)
            if file_age > 30 * 24 * 60 * 60:  # Recompute every 30 days
                os.remove(embeddings_file)
        except Exception as e:
            st.warning(f"Could not check embedding file age: {e}")
    
    if os.path.exists(embeddings_file):
        context_embeddings = torch.load(embeddings_file)
        print("Loaded pre-computed embeddings")
    else:
        print("Computing embeddings...")
        contexts = df['Context'].tolist()
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        torch.save(context_embeddings, embeddings_file)
        print("Saved embeddings to file")
    
    return context_embeddings

# Session State Initialization
def initialize_session_state():
    session_state_defaults = {
        'experiment_mode': False,
        'temperature': 0.4,
        'conversation_history': [],
        'hf_token': None,
        'client': None
    }
    
    for key, default_value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Load Dataset
@st.cache_data
def load_dataset(file_path):
    return pd.read_csv(file_path, low_memory=False)

# Find Most Similar Context
def find_most_similar_context(question, context_embeddings, contexts, responses):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return (
        contexts[most_similar_idx], 
        responses[most_similar_idx], 
        similarities[0][most_similar_idx].item()
    )
def initialize_inference_client(token):
    try:
        client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=token
        )
        
        # Debug: print available methods
        print("Available methods:")
        print(dir(client))
        
        return client
    except Exception as e:
        st.error(f"Client Initialization Error: {e}")
        return None
    
# Generate Model Response
def generate_model_response(client, prompt, temperature=0.4, max_tokens=500):
    """
    Generate response using Mistral model
    
    Args:
        client (InferenceClient): Hugging Face inference client
        prompt (str): Input prompt
        temperature (float): Response randomness
        max_tokens (int): Maximum response length
    
    Returns:
        str: Generated response
    """
    try:
        # Construct system and user message
        system_prompt = """You are an AI Powered Chatbot who provides remedies to queries. Your remedies should always be confident and emotionally supportive. 
        Focus on mental health and provide empathetic, actionable advice. choose the one with highest similarity score and do not show that in response"""
        
        # Combine system and user prompt
        full_prompt = f"{system_prompt}\n\nUser's message: {prompt}"
        
        # Use text_generation method
        response = client.text_generation(
            prompt=full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    except Exception as e:
        st.error(f"Response Generation Error: {e}")
        return "I'm experiencing some difficulties right now. Would you like to try again?"

def main():
    st.title("Large Language Models for Remedying Mental Status (LARMS)")
    
    # Initialize session state
    initialize_session_state()
    
    # Token Input Section
    if not st.session_state.hf_token:
        with st.form(key='token_form'):
            hf_token = st.text_input("Enter your Hugging Face Token:", type="password")
            submit_button = st.form_submit_button(label='Initialize Client')
            
            if submit_button and hf_token:
                # Try to initialize client with provided token
                client = initialize_inference_client(hf_token)
                if client:
                    st.session_state.hf_token = hf_token
                    st.session_state.client = client
                    st.success("Token validated and client initialized!")
                else:
                    st.error("Invalid token or unable to initialize client.")
    
    # If client is not initialized, show token input and exit
    if not st.session_state.client:
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Model Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused and deterministic."
        )
        st.session_state.temperature = temperature
        
        # Experiment mode toggle
        experiment_mode = st.toggle("Experiment Mode", value=st.session_state.experiment_mode)
        st.session_state.experiment_mode = experiment_mode
        
        # Display current settings
        st.write("Current Settings:")
        st.write(f"- Temperature: {st.session_state.temperature:.1f}")
        st.write(f"- Experiment Mode: {'On' if st.session_state.experiment_mode else 'Off'}")
    
    # Load dataset
    df = load_dataset('corpus/merged_dataset.csv')  # Update with your actual path
    contexts = df['Context'].tolist()
    responses = df['Response'].tolist()
    
    # Compute embeddings
    context_embeddings = load_or_compute_embeddings(df, embedding_model)
    
    # User input
    user_question = st.text_area("How are you feeling today?")
    
    if user_question:
        # Find most similar context
        with st.spinner("Finding the most similar context..."):
            similar_context, similar_response, similarity_score = find_most_similar_context(
                user_question, context_embeddings, contexts, responses
            )
        
        # Show experiment data if enabled
        if st.session_state.experiment_mode:
            with st.spinner("Loading experiment data..."):
                st.write("Similar Context:", similar_context)
                st.write("Suggested Response:", similar_response)
                st.write("Similarity Score:", f"{similarity_score:.4f}")
                st.write("Current Temperature:", f"{st.session_state.temperature:.1f}")
        
        # Construct prompt
        prompt = f"""You are an AI Powered Chatbot who provides remedies to queries. Your remedies should always be confident and emotionally supportive. 
        Focus on mental health and provide empathetic, actionable advice.

        User question: {user_question}
        Similar context from database: {similar_context}
        Suggested response: {similar_response}
        Similarity score: {similarity_score}

        {'EXPERIMENT MODE [Temp: ' + str(st.session_state.temperature) + '] - ' if st.session_state.experiment_mode else ''}
        Please provide a compassionate and helpful response."""
        
        # Generate response
        with st.spinner("Generating AI response..."):
            try:
                ai_response = generate_model_response(
                    st.session_state.client, 
                    prompt, 
                    temperature=st.session_state.temperature
                )
                
                # Display response
                st.text_area("AI's response:", value=ai_response, height=200, disabled=True)
                
                # Update conversation history
                st.session_state.conversation_history.append({"role": "user", "content": user_question})
                st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        # Display previous interactions
        if st.session_state.conversation_history:
            with st.expander("Show Previous Interactions"):
                for idx, interaction in enumerate(st.session_state.conversation_history):
                    if interaction['role'] == 'user':
                        st.markdown(f"**You:** {interaction['content']}")
                    else:
                        st.markdown(f"**LARMS:** {interaction['content']}")
                    if idx < len(st.session_state.conversation_history) - 1:
                        st.markdown("---")

if __name__ == "__main__":
    main()
