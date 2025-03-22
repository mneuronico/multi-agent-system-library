def tts(messages, manager):
    # First, get the 'response' field from the last message (from config.json, this will always be from 'talker')
    text = messages[-1].get("message").get("response")

    try:
        # Try Text To Speech, using a default voice and provider
        # You could get these parameters from the output of an agent or from a file
        # Not specifying the model will default to 'eleven_multilingual_v2'
        file_path = manager.tts(text, voice="sarah", provider = "elevenlabs")

        # Importantly, you need to return the "response" field for Telegram Integration to detect it automatically, and the value should be a dictionary with the 'voice_note' key for Telegram to send it as a voice note.
        return {"response": {"voice_note": file_path}}
    except:
        # If Text To Speech fails, just return the text as the response's value, so that Telegram will send it as a text message
        return {"response": text}