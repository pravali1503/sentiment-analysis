
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

MODEL_PATH = "pasagudula/sentiment-analysis-model"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #202c45);
    color: white;
}

h1 {
    text-align: center;
    color: #7dd3fc;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 18px;
}

.stButton > button {
    width: 100%;
    background-color: #0ea5e9;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][predicted_class].item()

    label = model.config.id2label.get(
        predicted_class,
        str(predicted_class)
    ).lower()

    label_map = {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive"
    }

    sentiment = label_map.get(label, label)

    return sentiment, confidence


st.title("😊 Sentiment Analysis")

st.markdown(
    '<p class="subtitle">Discover the sentiment behind your text</p>',
    unsafe_allow_html=True
)

st.write(
    "Enter a review or any text below to analyze its sentiment."
)

review = st.text_area(
    "Enter your text",
    placeholder="Example: I really love this product. It is amazing!",
    height=150
)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter some text first.")

    else:

        with st.spinner("Analyzing your text..."):
            sentiment, confidence = predict_sentiment(review)

        st.divider()
        st.subheader("Analysis Result")

        if sentiment == "positive":
            st.success("😊 Positive Sentiment")

        elif sentiment == "negative":
            st.error("😞 Negative Sentiment")

        elif sentiment == "neutral":
            st.info("😐 Neutral Sentiment")

        else:
            st.warning(f"Predicted Sentiment: {sentiment}")

        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.markdown("### Your Text")
        st.write(review)