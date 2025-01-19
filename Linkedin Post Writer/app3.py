from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
import streamlit as st
from phi.llm.groq import Groq
import os

# Set up the Streamlit app
st.title("AI LinkedIn Post Content Writer ✍️")
st.caption("Create professional and engaging LinkedIn posts with ease using AI LinkedIn Post Content Writer powered by Groq")

# Get Groq API key from user
groq_api_key = os.environ['GROQ_API_KEY']

# Get SerpAPI key from the user
serp_api_key = os.environ['SERPER_API_KEY']

if groq_api_key and serp_api_key:
    researcher = Assistant(
        name="Researcher",
        role="Searches for relevant content ideas, trends, and best practices for LinkedIn posts based on user preferences",
        llm=Groq(id="mixtral-8x7b-32768"),
        description=dedent(
            """\
        You are a world-class content researcher. Given a LinkedIn post topic and style preferences, generate a list of relevant content ideas, industry trends, and best practices for creating effective LinkedIn posts.
        Then search the web for each content idea, analyze the results, and return the 10 most relevant insights for creating engaging LinkedIn posts.
        """
        ),
        instructions=[
            "Given a LinkedIn post topic and style preferences, first generate a list of 3 search terms related to that topic.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant insights, trends, and best practices for creating LinkedIn posts.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )

    writer = Assistant(
        name="Writer",
        role="Generates a compelling LinkedIn post based on user preferences, research insights, and best practices",
        llm=Groq(id="mixtral-8x7b-32768"),
        description=dedent(
            """\
        You are an expert LinkedIn content writer. Given a LinkedIn post topic, style preferences, and a list of research insights,
        your goal is to create a professional, engaging, and impactful LinkedIn post that resonates with the target audience.
        """
        ),
        instructions=[
            "Given a LinkedIn post topic, style preferences, and a list of relevant content research insights, generate a compelling LinkedIn post.",
            "Ensure the LinkedIn post is well-structured, professional, and attention-grabbing.",
            "The tone should match the user's preferred writing style (e.g., casual, professional, inspiring).",
            "Incorporate relevant industry trends, facts, and insights into the post.",
            "Focus on clarity, engagement, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

    # Input fields for the user's LinkedIn post topic and style preferences
    post_topic = st.text_input("What is the topic of your LinkedIn post?")
    style_preference = st.text_input("What style would you like the LinkedIn post to be written in? (e.g., casual, professional, inspiring)")

    if st.button("Generate LinkedIn Post"):
        with st.spinner("Processing..."):
            # Get the research results
            research_results = researcher.run(f"Linkedin Post topic: {post_topic} for the style preference: {style_preference}", stream=False)

            # Generate the Linkedin Post Content
            response = writer.run(f"LinkedIn post on '{post_topic}' with style '{style_preference}' using the following research:\n\n{research_results}", stream=False)
            st.write("Linkedin Post Content")
            st.write(response)

