from textwrap import dedent
from phi.assistant import Assistant
import streamlit as st
from phi.llm.openai import OpenAIChat
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Set up the Streamlit app
st.title("YouTube Video Summarizer 🎥")
st.caption("Summarize YouTube videos in a more insightful and detailed manner!")

# Get OpenAI API key from user
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the OpenAI API if the key is available
if openai_api_key:
    # First agent to fetch video captions
    caption_fetcher = Assistant(
        name="CaptionFetcher",
        role="Fetches captions from YouTube videos",
        llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
        description=dedent(
            """\
        You are an AI that fetches captions from YouTube videos. Given a YouTube video URL, fetch its captions for further analysis.
        """
        ),
        instructions=[
            "Fetch the captions from the given YouTube video URL.",
        ],
        add_datetime_to_instructions=True,
    )

    # Second agent to summarize the captions with a focus on quality and details
    summarizer = Assistant(
        name="Summarizer",
        role="Summarizes YouTube video captions in detail",
        llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
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
    )

    # Input field for YouTube video URL
    video_url = st.text_input("Enter YouTube video URL:")

    def get_video_id(url):
        # Extract video ID from YouTube URL
        if "youtube.com/watch?v=" in url:
            return url.split("v=")[-1]
        return None

    def fetch_captions(video_id, languages=["en", "hi"]):
        # Fetch captions using youtube-transcript-api (try both English and Hindi)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            # Return the transcript in a simple string format
            caption_text = "\n".join([entry['text'] for entry in transcript])
            return caption_text
        except TranscriptsDisabled:
            st.error("Captions are disabled for this video.")
            return None
        except NoTranscriptFound:
            st.error("No transcript found for this video in the requested languages.")
            return None
        except Exception as e:
            st.error(f"Error fetching captions: {str(e)}")
            return None

    if st.button("Summarize Video"):
        if video_url:
            video_id = get_video_id(video_url)
            if video_id:
                with st.spinner("Fetching captions and summarizing..."):
                    # Fetch captions in English or Hindi
                    captions = fetch_captions(video_id)

                    if captions:
                        # Pass the captions to the second agent (Summarizer) for summarization
                        summary = summarizer.run(captions, stream=False)
                        st.write(summary)
                    else:
                        st.error("Could not fetch captions for this video.")
            else:
                st.error("Invalid YouTube video URL.")
        else:
            st.error("Please enter a valid YouTube video URL.")
