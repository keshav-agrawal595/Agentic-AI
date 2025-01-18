from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
import streamlit as st
from phi.llm.google import Gemini
import os

# Set up the Streamlit app
st.title("AI Blog Writer 📝")
st.caption("Generate creative and engaging blog posts with the AI Blog Writer using Gemini")

# Get Groq API key from user
gemini_api_key = os.environ['GEMINI_API_KEY']

# Get SerpAPI key from the user
serp_api_key = os.environ['SERPER_API_KEY']

if gemini_api_key and serp_api_key:
    researcher = Assistant(
        name="Researcher",
        role="Searches for blog topics, ideas, and content inspiration based on user preferences",
        llm=Gemini(id="gemini-1.5-flash"),
        description=dedent(
            """\
        You are a world-class content researcher. Given a blog topic and writing style preferences, generate a list of search terms for finding relevant content ideas, trends, and research.
        Then search the web for each term, analyze the results, and return the 10 most relevant content ideas.
        """
        ),
        instructions=[ 
            "Given a blog topic and style preferences, first generate a list of 3 search terms related to the topic and preferences.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant content ideas to the user's preferences.",
            "Remember: the quality of the content ideas is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    writer = Assistant(
        name="Writer",
        role="Generates a draft blog post based on user preferences and research results",
        llm=Gemini(id="gemini-1.5-flash"),
        description=dedent(
            """\
        You are an expert blog writer. Given a blog topic, style preferences, and a list of content research results,
        your goal is to generate a full and engaging blog post that aligns with the user's style and incorporates relevant research.
        """
        ),
        instructions=[
            "Given a blog topic, style preferences, and a list of content research results, generate a full blog post that includes relevant ideas, examples, and insights.",
            "Ensure the blog post is well-structured, engaging, and informative from introduction to conclusion.",
            "The tone should match the user's preferred writing style (e.g., casual, professional, informative).",
            "Avoid asking the user for additional details; the blog post should be created with the information already provided.",
            "Focus on creativity, clarity, and a compelling narrative.",
            "Provide relevant facts, examples, and research without making up details.",
        ],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

    # Input fields for the user's blog topic and style preferences
    blog_topic = st.text_input("What blog topic are you writing about?")
    style_preference = st.text_input("What style would you like the blog post to be written in? (e.g., casual, professional, informative)")

    if st.button("Generate Blog Post"):
        with st.spinner("Processing..."):
            # Get the response from the assistant
            response = writer.run(f"Blog on {blog_topic} with style {style_preference}", stream=False)
            st.write(response)
