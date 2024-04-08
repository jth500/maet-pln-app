import streamlit as st


def sidebar():
    with st.sidebar:
        st.title("The App")
        st.write(
            "Explore the summaries produced by our models trained with 1) "
            "Reinforcement Learning with AI Feedback (RLAIF), and 2) Supervised Fine Tuning (SFT) by switching between the tabs."
        )
        st.title("Methodology")
        st.write(
            """In this project, we investigate whether performing RLAIF on small """
            """Large Language Models (LLM) can bridge the gap with larger "off-the-shelf" LLMs. We use """
            """GPT-2 Medium and T5-base."""
        )
        st.write(
            """**Supervised Fine Tuning** for summarization adjusts a pre-trained model using a """
            """dataset of text-summary pairs to improve summary generation. This method refines """
            """the model’s understanding of summarization tasks, enhancing its ability to produce """
            """concise and accurate summaries for specific domains."""
        )
        st.write(
            """**Reinforcement Learning** is an area of machine learning where an """
            """agent learns to make decisions by performing actions in an environment """
            """to achieve some goal. The agent receives feedback through rewards or """
            """penalties based on the outcomes of its actions. In RLAIF, human feedback is replaced with """
            """feedback from another agent."""
        )
        st.write("Read the full report here **soon**.")


def info_expander():
    with st.expander("Built by maet-pln for COMP0087"):
        st.write(
            "This app was built as part of the assessment for COMP0087 "
            "(Statistical Natural Language Processing) at University College London 23/24."
        )
        st.write(
            "Our team consists of [Isaac](https://github.com/ijwatson98), [Toby](https://github.com/jth500), [Lucia](https://github.com/guillametlucia), [Jack](https://www.linkedin.com/in/jackwardleprofile?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), [Louise](https://www.linkedin.com/in/louise-sandland-b2600b207/). We are a group "
            "of MSc students enrolled in 1) Computational Statistics and Machine Learning and 2) Data Science."
        )
        st.write(
            "The codebase for training our models is [here](https://github.com/jth500/maet-pln), and the code for this app is [here](https://github.com/jth500/maet-pln-app)."
        )
