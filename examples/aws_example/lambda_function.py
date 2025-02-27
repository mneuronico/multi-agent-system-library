import json
import boto3
import os
import requests
import mas
from mas import AgentSystemManager

# Replace with your bot token
BUCKET_NAME = "ada-bot-data"

lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')

# S3 key template for sqlite history files (each file named <chat_id>.sqlite inside "history/" folder)
def s3_sqlite_key(chat_id):
    return f"history/{chat_id}.sqlite"

def lambda_handler(event, context):
    try:
        # Check if this is the initial invocation from Telegram
        if 'body' in event:

            print("----- INITIAL INVOCATION ----- NOW WITH MAS")
            print("MAS VERSION:", mas.__version__)

            body = json.loads(event['body'])
            chat_id = body['message']['chat']['id']
            text = body['message']['text']

            print("TEXT:", text)

            # Acknowledge the webhook immediately
            response = {
                "statusCode": 200,
                "body": json.dumps({"status": "Message received"})
            }

            # Invoke the same Lambda function asynchronously
            payload = {
                "chat_id": chat_id,
                "text": text
            }
            lambda_client.invoke(
                FunctionName=context.invoked_function_arn,
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps(payload)
            )

            print("LAMBDA INVOKED, RETURNING RESPONSE TO TELEGRAM...")

            return response
        else:
            # This is the asynchronous invocation for processing
            print("----- ASYNCHRONOUS INVOCATION ----- THIS IS WWL")

            chat_id = event['chat_id']
            text = event['text']
            process_message(chat_id, text)
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "Message processed"})
            }
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    
def process_message(chat_id, text):
    print("----- PROCESSING MESSAGE... -----")

    try:
        manager = AgentSystemManager(config_json="config.json",
                                     history_folder="/tmp/history",
                                     files_folder="/tmp/files"
        )

    except Exception as e:
        print("Issue while creating the manager:", e)
        return

    # Build the S3 key for the sqlite file for this chat_id
    key = s3_sqlite_key(chat_id)

    try:
        # Try to download the sqlite file from S3
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        sqlite_bytes = response['Body'].read()
        # Import the downloaded sqlite file into the manager (it will be saved in /tmp/history/<chat_id>.sqlite)
        manager.import_history(chat_id, sqlite_bytes)
    except s3_client.exceptions.NoSuchKey:
        # No history exists for this chat_id; a new DB will be created when needed
        print(f"No existing history for user {chat_id} found in S3.")
    except Exception as e:
        print("Error retrieving sqlite from S3:", e)

    try:
        # Set the current user in the manager so that history operations use the correct DB file
        manager.set_current_user(chat_id)

        # Run the agent on the input text
        output = manager.run(
            input=text,
            verbose=True
        )
        
        print("Received output from manager:",output)
    except Exception as e:
        print("Issue while getting response:", e)
        output = {"response": "Sorry, I made a mistake :("}

    try:
        TELEGRAM_BOT_TOKEN = manager.get_key("TELEGRAM_TOKEN")
        # Send the output response back to Telegram
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": output.get("response", "No response")
        }
        requests.post(telegram_url, json=payload)
    except Exception as e:
        print("Issue while sending message to Telegram:", e)

    try:
        # Export the updated sqlite file from the manager and upload it to S3
        new_sqlite_bytes = manager.export_history(chat_id)
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=new_sqlite_bytes)
    except Exception as e:
        print("Issue while exporting history:", e)