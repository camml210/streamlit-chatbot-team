import streamlit as st
 
def main():
    st.title('Chatbot Team')
    st.header('Login')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')
    
if st.button('Login'):
        st.success('Login successful')
        st.subheader('Chatroom')
        
if __name__ == '__main__':
    main()