import streamlit as st
from chatbot import ChatbotTeam
from PIL import Image

def main():
    st.set_page_config(layout="wide")
    st.title("Chatbot Team Application")

    menu = ["Home", "Chatroom"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to our Chatbot Team Application!")

    elif choice == "Chatroom":
        st.subheader("Chatroom")
        st.info("Please login to access the chatroom")

        chatroom = Chatroom()
        chatroom.chat_interface()

class Chatroom:
    def __init__(self):
        self.chatbot_team = ChatbotTeam()
        self.agents = []  # List to store agents

    def chat_interface(self):
        # Set page layout
        st.set_page_config(layout="wide")

        # Set chatroom background image
        background_image = Image.open("chatroom_bg.jpg")
        st.image(background_image, use_column_width=True)

        # Add chatroom title
        st.markdown("<h1 style='text-align: center; color: white;'>Chatroom</h1>", unsafe_allow_html=True)

        # Add chat messages container
        chat_container = st.empty()
        chat_messages = []

        # Add input message box
        input_message = st.text_input("User Input")

        # Add send button
        if st.button("Send"):
            if input_message:
                # Process user input and get response
                response = self.chatbot_team.run(input_message)

                # Add user input to chat messages
                chat_messages.append(("User", input_message))

                # Add agent responses to chat messages
                for agent, message in response.items():
                    chat_messages.append((agent, message))

                # Clear input message
                input_message = ""

        # Render chat messages
        for agent, message in chat_messages:
            if agent == "User":
                # Display user message on the left
                chat_container.markdown(f"<p style='text-align: left; color: white;'>{message}</p>", unsafe_allow_html=True)
            else:
                # Display agent message on the right
                chat_container.markdown(f"<p style='text-align: right; color: white;'>{agent}: {message}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
