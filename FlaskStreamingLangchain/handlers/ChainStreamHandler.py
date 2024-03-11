from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    """A callback handler for streaming tokens."""
    
    def __init__(self, gen):
        """Initialize the ChainStreamHandler with a generator."""
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        """Callback function called when a new token is generated."""
        print("Generated token:", token) 
        self.gen.send(token)
