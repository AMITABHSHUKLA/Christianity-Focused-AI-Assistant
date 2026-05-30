import streamlit as st
from core_logic import (
    load_and_index_bible, 
    check_moderation, 
    process_image_request, 
    process_text_request
)

st.set_page_config(page_title="Christian AI Assistant", layout="wide")
st.title("Christianity-Focused AI Assistant")

# gotta init session state so memory doesn't wipe when ui reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag_system():
    return load_and_index_bible()

demo_chunks, vector_index, bm25_index = init_rag_system()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a theological question or request an image..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            
            # top layer gatekeeper
            if check_moderation(prompt):
                response_text = "I cannot process this request due to strict safety and moderation policies."
                st.markdown(response_text)
            
            # routing visual intent
            elif any(word in prompt.lower() for word in ["image", "draw", "picture"]):
                img_url, msg = process_image_request(prompt)
                if img_url:
                    st.image(img_url, caption=msg)
                    response_text = "Here is your generated image."
                else:
                    response_text = msg
                    st.markdown(response_text)
            
            # standard rag route
            else:
                response_text = process_text_request(
                    prompt, 
                    st.session_state.messages, 
                    demo_chunks, 
                    vector_index, 
                    bm25_index
                )
                st.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})