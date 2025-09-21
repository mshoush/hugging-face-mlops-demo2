import torch
from transformers import pipeline
import gradio as gr

# load summarization model
model = pipeline("summarization", model="facebook/bart-large-cnn")  # specify model

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

# create Gradio interface
with gr.Interface(fn=predict, inputs="textbox", outputs="text") as interface:
    interface.launch()



# from transformers import pipeline
# import gradio as gr


# model = pipeline(
#     "summarization",
# )

# def predict(prompt):
#     summary = model(prompt)[0]["summary_text"]
#     return summary


# # create an interface for the model
# with gr.Interface(predict, "textbox", "text") as interface:
#     interface.launch()