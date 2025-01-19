from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
from phi.llm.openai import OpenAIChat
import streamlit as st
import os

# Set up the Streamlit app
st.title("AI Blog Writer ✍️")
st.caption("Create research-backed and engaging blogs using AI.")

# Get API keys
openai_api_key = os.environ['OPENAI_API_KEY']
serp_api_key = os.environ['SERPER_API_KEY']

if openai_api_key and serp_api_key:
    researcher = Assistant(
        name="Researcher",
        role="Conducts research for blog topics and gathers information",
        llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
        description=dedent(
            """\
            You are an expert researcher. Given a blog topic, generate a list of search terms, research the topic, 
            and return 10 high-quality references or insights for the blog.
            """
        ),
        instructions=[
            "Generate 3 search terms for the given blog topic.",
            "Use `search_google` to gather data for these terms.",
            "Analyze the results and return the 10 most relevant insights or references.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
    )

    writer = Assistant(
        name="Writer",
        role="Drafts a blog post based on research",
        llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
        description=dedent(
            """\
            You are a professional blog writer. Use the research data to draft a blog that is engaging, structured, 
            and optimized for the target audience.
            """
        ),
        instructions=[
            "Draft a blog post using the research insights provided.",
            "Ensure the blog has an introduction, structured body, and conclusion.",
            "Include a CTA and naturally integrate relevant keywords for SEO.",
        ],
    )

    # Input fields for the blog topic and target audience
    topic = st.text_input("What is the blog topic?")
    audience = st.text_input("Who is the target audience?")

    if st.button("Generate Blog"):
        with st.spinner("Researching and writing..."):
            # Get the research results
            research_results = researcher.run(f"Research blog topic: {topic} for the audience: {audience}", stream=False)
            
            # Generate the blog post
            blog = writer.run(
                f"Write a blog on the topic '{topic}' for the audience '{audience}' using the following research:\n\n{research_results}",
                stream=False,
            )
            st.write("### Generated Blog:")
            st.write(blog)

