# import necessary dependencies
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompts import contextualize_prompt,answer_prompt
from src.history import get_session_history


# format retrieved docs, preserves the page information
def format_docs(docs):
    formated =[]
    for doc in docs:
        page = doc.metadata.get(
            "page",
            'N/A'
        )

        formated.append(
            f"page: {page}\n"
            f'{doc.page_content}'
        )
    return "\n\n".join(formated)

# function to build the converstational rag chain
def build_rag_chain(retriever,
                    temperature=0.0):
    # intialize the llm model
    llm= ChatGroq(
        model='qwen/qwen3.8-27b',
        temperature=0,
        max_tokens=500
    )

# question rewriter chain
    question_rewriter = (contextualize_prompt
                      | llm
                      | StrOutputParser()
    )
    # build the answer chain
    answer_chain = (
                answer_prompt
                | llm
                | StrOutputParser()
    )
    
    # function to run the rag chain
    def run_rag(inputs):

        question = inputs['question']
        history = inputs.get("chat_history",[])

        # create the standalone question using question_rewriter chain
        if history:
            standalone_question = (
                question_rewriter.invoke({
                    "chat_history":history,
                    "question":question
                })
            )
        # First question already has no prior
        # conversational dependency.
        else: 
            standalone_question = question


        # then retrieve the relevent documnets for standalone question
        docs = retriever.invoke(standalone_question)

        # create the context
        context = format_docs(docs)


        # invoke the answer_chain
        answer = answer_chain.invoke(
            {
                "context":context,
                "chat_history":history,
                "question": question
            }
        )
        # return the answer and relevent retrieved docs
        return {
            'answer':answer,
            'sources':docs,
            'original_question':question,
            'standalone_question': standalone_question
        }

    # intialize the runnable lambda with run_rag function
    rag = RunnableLambda(run_rag)

    # finally return the runnables
    converataion= RunnableWithMessageHistory(
        rag,
        get_session_history,
        input_messages_key='question',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )
    return converataion 




