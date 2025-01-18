from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
import streamlit as st
from phi.llm.groq import Groq
import os

# Set up the Streamlit app
st.title("AI Blog Writer ✍️")
st.caption("Create research-backed and engaging blogs using Groq AI and SerpAPI.")

# Get Groq API key from user
groq_api_key = os.environ['GROQ_API_KEY']

# Get SerpAPI key from the user
serp_api_key = os.environ['SERPER_API_KEY']

if groq_api_key and serp_api_key:
    researcher = Assistant(
        name="Researcher",
        role="Searches for blog topic-related information and generates relevant references",
        llm=Groq(id="llama-3.3-70b-versatile"),
        description=dedent(
            """\
        You are a world-class blog researcher. Given a blog topic and target audience, generate a list of search terms for finding relevant articles, research papers, and other resources.
        Then search the web for each term, analyze the results, and return the 10 most relevant insights.
        """
        ),
        instructions=[
            "Given a blog topic and target audience, first generate a list of 3 search terms related to that topic and the audience.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant insights or references to inform the blog post.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    writer = Assistant(
        name="Writer",
        role="Generates a draft blog post based on the research results",
        llm=Groq(id="llama-3.3-70b-versatile"),
        description=dedent(
            """\
        You are a professional blog writer. Given a topic, audience, and research results, generate a well-structured, engaging blog post.
        Ensure the blog post includes an introduction, body, conclusion, and a call-to-action (CTA).
        """
        ),
        instructions=[
            "Given a blog topic, target audience, and a list of research insights, draft a blog post.",
            "Ensure the blog post is well-structured, informative, and engaging.",
            "Include a clear introduction, body, and conclusion, and end with a call-to-action (CTA).",
            "Remember: the quality of the blog is paramount, with correct citations and integration of insights.",
        ],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

    # Input fields for the user's blog topic and target audience
    topic = st.text_input("Enter the blog topic:")
    audience = st.text_input("Who is the target audience?")

    if st.button("Generate Blog"):
        with st.spinner("Researching and writing..."):
            # Get the research results
            research_results = researcher.run(f"Research blog topic: {topic} for the audience: {audience}", stream=False)
            st.write("### Research Results:")
            st.write(research_results)

            # Generate the blog post
            blog = writer.run(
                f"Write a blog on the topic '{topic}' for the audience '{audience}' using the following research:\n\n{research_results}",
                stream=False,
            )
            st.write("### Generated Blog:")
            st.write(blog)
