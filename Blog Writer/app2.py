from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
import streamlit as st
import os
from phi.model.google import Gemini

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
        role="Searches for blog topic-related information and generates relevant references",
        model=Gemini(id="gemini-1.5-flash"),
        description=dedent(
            """\
            You are a world-class blog researcher. Given a blog topic and target audience, generate a list of search terms for finding relevant articles, research papers, and other resources.
            Then search the web for each term, analyze the results, and return the 10 most relevant insights.
            """
        ),
        instructions=[
            "Generate a list of 3 search terms related to the topic and audience.",
            "Search the web for each term using the SerpAPI tool.",
            "Analyze the results and return the 10 most relevant insights or references.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
    )

    writer = Assistant(
        name="Writer",
        role="Generates a draft blog post based on user preferences and research results",
        model=Gemini(id="gemini-1.5-flash"),
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
        show_tool_calls=True,
        num_history_messages=3,
        markdown=True,
    )

    # Input fields for the user's blog topic and target audience
    topic = st.text_input("Enter the blog topic:")
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

