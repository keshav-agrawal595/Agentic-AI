from textwrap import dedent
from phi.assistant import Assistant
import streamlit as st
from phi.llm.groq import Groq
import os
from phi.tools.youtube_tools import YouTubeTools
from phi.tools.duckduckgo import DuckDuckGo

# Set up the Streamlit app
st.title("YouTube Video Summarizer 🎥")
st.caption("Summarize YouTube videos in a detailed and insightful way!")

# Get Groq API key from user
groq_api_key = os.environ['GROQ_API_KEY']

if groq_api_key:
    # First agent to fetch video captions
    caption_fetcher = Assistant(
        name="CaptionFetcher",
        role="Fetches captions from YouTube videos",
        llm=Groq(id="llama-3.3-70b-versatile"),
        description=dedent(
            """\
        You are a Youtube Agent that fetches captions from YouTube videos. Given a YouTube video URL, fetch its captions for further analysis. 
        The captions can be in any language. Don't ask for any confirmation, just give me the captions.
        """
        ),
        instructions=[
            "No matter what is the captions langauge, Fetch the captions from the given YouTube video URL.",
        ],
        tools=[YouTubeTools(),DuckDuckGo()],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        get_video_captions = True,
    )

    # Second agent to summarize the captions with a focus on quality and details
    summarizer = Assistant(
        name="Summarizer",
        role="Summarizes YouTube video captions in detail",
        llm=Groq(id="llama-3.3-70b-versatile"),
        description=dedent(
            """\
        You are an AI that summarizes YouTube video captions in a detailed and insightful way. Given the captions from a YouTube video, 
        provide a detailed summary that includes the main ideas, key points, and insights from the video. Structure the summary to make it 
        informative, covering all key elements and giving a clear overview of the video's content. Focus on making the summary easy to understand 
        while providing enough depth to convey the value of the video.
        """
        ),
        instructions=[
            "Analyze the captions from the YouTube video and summarize the main ideas and key points in a detailed manner.",
            "Ensure that the summary includes any important insights, examples, or lessons from the video.",
            "Provide a structured summary that highlights the main themes and conclusions.",
            "Focus on clarity, coherence, and detail in your summary.",
        ],
        add_datetime_to_instructions=True,
        tools=[YouTubeTools(),DuckDuckGo()],
        show_tool_calls=True,
        get_video_captions = True,
    )

    # Input field for YouTube video URL
    video_url = st.text_input("Enter YouTube video URL:")

    if st.button("Summarize Video"):
            with st.spinner("Fetching captions and summarizing..."):
                caption_results = caption_fetcher.run(f"Youtube Video Link : {video_url}", stream=False)

                # Pass the captions to the second agent (Summarizer) for summarization
                summary = summarizer.run(f"Summarize the youtube video : {video_url} using the following caption data of the video : \n\n{caption_results}", stream=False)
                st.write(summary)
