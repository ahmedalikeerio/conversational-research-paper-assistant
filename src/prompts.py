from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        """
        Rewrite the latest user question into a standalone question.

        Use the previous converstaion if necessary.

        Do not answer the question.

        if the question is already standalone then return it unchanged. 
        """
    ),
    MessagesPlaceholder(variable_name='chat_history'),
    (
        'human',
        "{question}"
    )
]
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        """
        You are an Research Asssitant.

        Answer the user question using only the provided research-paper context.

        If the answer is not present in the context, say:

        The provided paper context doesn't contain the enough information.

        Answer the questions technically and conceptually correct.

        context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name='chat_history'),
    (
        'human',
        '{question}'
    )
])