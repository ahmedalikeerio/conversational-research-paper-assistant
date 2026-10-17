from langchain_core.chat_history import InMemoryChatMessageHistory

# creating the empty store
store = {}

# function to get the session hitsory using session_id
def get_session_history(session_id):
    # checking if session_id is not stored in the store
    if session_id not in store:
        # save the chat memory in store
        store[session_id] = InMemoryChatMessageHistory()

    # returuning the session id
    return store[session_id]