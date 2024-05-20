from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
#import transformers.utils
import streamlit as st


template = """
You are a poet.
Please write a poem about a {topic} a user entered.
Don't say anything other than the poem you wrote. Just print out the poem.
Please write a poem in 100 characters.
Add \n at the end of each sentence
Please use emojis a lot.
"""

prompt = PromptTemplate(template=template, input_variables=["topic"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path = "llama-2-7b-chat.Q2_k.gguf",
    input={"temperature": 0,
           "max_length": 30,
           "top_p": 1},
        callback_manager=callback_manager,
        verbose=True,
)

# llm = HuggingFacePipeline.from_model_id(model_id="beomi/KoAlpaca-Polyglot-5.8B",
#                                         task="text-generation",
#                                         model_kwargs={"do_sample" : True,
#                                                       "temperature":0.7,
#                                                       "max_length":2048,
#                                                       "torch_dtype":torch.float32},
#                                         device=-1) #cpu -1

llm_chain = LLMChain(prompt=prompt, llm=llm)


st.title('My Poet :umbrella_with_rain_drops:')

with st.chat_message("AI"):
    st.write("Thanks for coming:blush: Tell me the topic.")
    st.write("I'll write you a nice poem:crystal_ball:")

input = st.chat_input(placeholder="Please enter a topic.")
topic = str(input)
if input:
    with st.chat_message("user"):
        st.write(topic)

prompt = "Please write a poem about the" + topic

if input:
    with st.chat_message("AI"):
        with st.spinner('Wait for it...'):
            st.write(llm_chain.run(prompt))