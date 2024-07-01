import streamlit as st
import fandom

from tools.fandom_tools import search_tool, get_subsections_tool, summarize_tool
from tools.general_tools import roll_dice_tool

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts.chat import PromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Streamlit app
st.title("RPG Dungeon Master")

# Ask the user for the fandom if not already selected
if "fandom_input" not in st.session_state:
    fandom_input = st.text_input("Please enter the fandom you want to use:")
    st.write("Please enter a fandom to continue.")
    if fandom_input:
        st.session_state.fandom_input = fandom_input.capitalize()
        fandom.set_wiki(fandom_input)
        st.rerun()
else:
    st.text_input("Please enter the fandom you want to use:", value=st.session_state.fandom_input, disabled=True)

    if "messages" not in st.session_state:
        # Import the greeting message
        with open("prompts/greeting.txt", "r") as file:
            greeting_message = file.read()

        st.session_state.messages = [{"role": "assistant", "content": greeting_message.format(fandom_name=st.session_state.fandom_input)}]
    
    # Initialize the agent if not already initialized
    if "agent" not in st.session_state:
        # Initialize the LLM and tools
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        tools = [search_tool, get_subsections_tool, summarize_tool, roll_dice_tool]

        # Import the system prompt
        with open("prompts/system_prompt.txt", "r") as file:
            system_prompt = file.read()

        # Define the prompt for the agent
        prompt = (SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt.format(fandom_name=st.session_state.fandom_input)))
                + MessagesPlaceholder(variable_name='chat_history', optional=True)
                + HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))
                + MessagesPlaceholder(variable_name='agent_scratchpad'))

        # Create the agent and agent executor
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        st.session_state.agent = agent_executor

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_input := st.chat_input():
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.agent.invoke(
                {"input": user_input, "chat_history": st.session_state.messages}, {"callbacks": [st_callback]}
            )
            assistant_output = response["output"]
            st.markdown(assistant_output)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_output})