import streamlit as st
from  youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from generator_ai_youtube_app import create_vectorstore, fetch_youtube_transcript, generate_response, split_transcript


# setting page config
st.set_page_config(
    page_title= " VidSynth AI",
    page_icon="📺",
    layout = "wide"
)

api = YouTubeTranscriptApi()

# error logging
ERROR_KEY_WORDS = ["Error", "Resource_exhausted", "API Key Not found", "QUOTA_EXHAUSTED"]

def has_error(content):
    return any(keyword.lower() in content.lower() for keyword in ERROR_KEY_WORDS)

def show_error_message(content, context):
    if has_error(content):
        st.error(f"Found an error while fetching {context} - 💳 Add billing at console.cloud.google.com  - 🔑 Create a new API key at aistudio.google.com ")
        return True
    return False

st.title(" YouTube Content Synthesizer")
st.subheader(" Paste a video link and select a task from the sidebar")
st.divider()

options_list = ["Chat with Video", "Notes For You"]

# creating a sidebar
with st.sidebar:
    st.header("📺 VidSynth AI ")
    st.markdown("--------------------")
    st.subheader(" Transform any Youtube video into key topics a podcast, or a chatbot.")
    user_input = st.sidebar.text_input("YouTube URL",placeholder="Enter your YouTube URL here")
    video_lang_code = st.sidebar.text_input("Video Language Code",placeholder="Enter your language code")

    genre = st.radio(
        label = "Choose what you want to generate",
        options = options_list,
        index =0 # sets the default choice 
    )
    st.write(user_input, video_lang_code)
    processing_button = st.button(" 🖼️ Start Processing")


def get_youtube_id(url):
    """
    it will gett the normal youtube url and parse it to get the id
    """
    #parse the url 
    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtube.de":
        return parsed_url.path[1:]
    
    if parsed_url.hostname in ("www.youtube.com",'youtube.de','youtube.com','m.youtube.com'):
        query_params= parse_qs(parsed_url.query)
        return query_params.get('v',[None])[0]
    
    return None

# initialize session state
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Chat with Video ──────────────────────────────────────────────────────────
if genre == "Chat with Video":
    st.info(f"Selected option: {genre}")

    if processing_button:
        if not user_input:
            st.warning("Please enter a YouTube URL to proceed.")
        else:
            yt_video_id = get_youtube_id(user_input)

            if not yt_video_id:
                st.warning("Please enter a valid YouTube URL to proceed.")
            else:
                lang = [video_lang_code] if video_lang_code.strip() else ["en"]

                with st.spinner("Fetching the transcript... This may take a moment."):
                    try:
                        transcript = fetch_youtube_transcript(yt_video_id, lang=lang)
                    except Exception as e:
                        st.error(f"Error while fetching transcript: {e}")
                        st.stop()

                with st.spinner("Splitting the transcript... This may take a moment."):
                    try:
                        small_chunks = split_transcript(transcript)
                    except Exception as e:
                        st.error(f"Error while splitting transcript: {e}")
                        st.stop()

                with st.spinner("Building the vectorstore... This may take a moment."):
                    try:
                        st.session_state["vectorstore"] = create_vectorstore(small_chunks)
                        st.session_state["chat_history"] = []  # reset on new video
                    except Exception as e:
                        st.error(f"Error while building vectorstore: {e}")
                        st.stop()

                st.success("✅ Ready! Ask your questions below 🎉")

    # -- Chat UI (renders after processing, persists across reruns) --
    if st.session_state["vectorstore"] is not None:

        # render existing chat history
        for chat in st.session_state["chat_history"]:
            avatar = "🤖" if chat["role"] == "user" else "👾"
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"])

        # chat input must be OUTSIDE the for loop
        question = st.chat_input("Ask a question about the video:")
        if question:
            # display and store user message
            with st.chat_message("user", avatar="🤖"):
                st.markdown(question)
            st.session_state["chat_history"].append({"role": "user", "content": question})

            # generate and display assistant response
            with st.chat_message("assistant", avatar="👾"):
                with st.spinner("Generating response..."):
                    docs = st.session_state["vectorstore"].similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    response = generate_response(context, question)
                    st.markdown(response)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    else:
        st.info("👈 Enter a YouTube URL and click **Start Processing** to begin.")


# ── Notes For You ────────────────────────────────────────────────────────────
elif genre == "Notes For You":
    st.info(f"Selected option: {genre}")

    if processing_button:
        if not user_input:
            st.warning("Please enter a YouTube URL to proceed.")
        else:
            yt_video_id = get_youtube_id(user_input)

            if not yt_video_id:
                st.warning("Please enter a valid YouTube URL.")
            else:
                with st.spinner("Fetching the transcript..."):
                    try:
                        transcript = fetch_youtube_transcript(yt_video_id, lang=["en"])
                        st.subheader("📝 Video Transcript")
                        st.write(transcript)
                    except Exception as e:
                        st.error(f"Error fetching transcript: {e}")
    else:
        st.info("👈 Enter a YouTube URL and click **Start Processing** to begin.")