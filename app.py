import gradio as gr

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.llms import HuggingFaceHub

from langchain.embeddings import HuggingFaceHubEmbeddings

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA



def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, repo_id):
    
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":250})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

    return result

# CSS
css = """
#col-container {
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}

.title {
  text-align: center;
  max-width: 700px;
  color: #000;
}

.pdf-doc {
  margin-bottom: 10px;
}

.chatbot {
  height: 350px;
  border: 1px solid #ccc;
  padding: 10px;
  background-color: #fff;
  font-family: sans-serif;
  font-size: 16px;
  line-height: 24px;
}

.chatbot .message {
  color: #000;
}

.chatbot .user-message {
  background-color: #eee;
}

.chatbot .bot-message {
  background-color: #ccc;
}
"""

# HTML
title = """
<div style="text-align: center;max-width: 800px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .pdf from local machine, click the "Load PDF🚀" button, <br />
    When ready, you are all set to start asking questions from the pdf</p>
</div>
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column(elem_id="col-container"):
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            repo_id = gr.Dropdown(label="LLM", choices=["google/flan-ul2", "OpenAssistant/oasst-sft-1-pythia-12b", "bigscience/bloomz", "meta-llama/Llama-2-7b-chat-hf"], value="google/flan-ul2")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your Question and hit Enter ",elem_id="chatbot .user-message")
        submit_btn = gr.Button("Send message")
    #load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch(True)