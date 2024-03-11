from flask import Flask, make_response, Response, request, stream_with_context, jsonify
from flask_cors import CORS
from privategpt.utils import private_gpt_chain
from hr.utils import hr_gpt_chain
from gpt4vision.utils import gpt4vision
from dalle.utils import generate_image
from config.config import ALLOWED_API_KEY,INSTRUMENTATION_KEY
from applicationinsights import TelemetryClient
from applicationinsights.flask.ext import AppInsights

@app.route('/hrgptstream', methods=['GET'])
def hrgptstream():
    """
    Endpoint to generate tokens based on a prompt for HR Documents

    Returns:
        Response: Response object containing generated tokens.

    """
    if authenticate_request():
        prompt = request.args.get('prompt', "What are our holidays for 2024?")
        tc.track_trace(f"Generating tokens for HR prompt: {prompt}")

        def generate_tokens():
            g = hr_gpt_chain(prompt)
            try:
                while True:
                    token = next(g)
                    yield token
            except StopIteration:
                pass
        return Response(stream_with_context(generate_tokens()), mimetype='text/event-stream')
    else:
        tc.track_trace("Unauthorized request.")
        return jsonify({"error": "Unauthorized. Invalid or missing API key."}), 401
