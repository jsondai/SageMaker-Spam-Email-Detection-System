import json
import boto3
from botocore.exceptions import ClientError
import logging
import os
from io import StringIO
import csv
from utils import *


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

AWS_REGION = "us-east-1"
s3 = boto3.client('s3')
sagemaker = boto3.client('runtime.sagemaker')
ENDPOINT_NAME = 'HW3-SpamDetector-API-Endpoint'
vocabulary_length = 9013

def lambda_handler(event, context):
    logger.debug(event)

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    data = s3.get_object(Bucket=bucket, Key=key)
    contents = data['Body'].read().decode()
    logger.debug("[***DEBUG***] Email Body before parsing: ")
    logger.debug(contents)

    body_list = contents.split("--")[1].split("\n\n")[0].split('\r\n\r\n')
    body = "".join(body_list[1:])
    # logger.debug(body)

    body = " ".join(body.splitlines()).replace("=","")
    logger.debug("[***DEBUG***] Email Body after parsing: ")
    logger.debug(body)

    test_messages = [body]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    # Run model endpoint w/ vect + encode
    io = StringIO()
    json.dump(encoded_test_messages.tolist(), io)
    response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                         ContentType='application/python-pickle',
                                         Body = bytes(io.getvalue(),'utf-8')  )
    print(response)
    result = response['Body'].read()
    print(result)
    result = json.loads(result)


    #### Return Msg

    SUBJECT = "AUTOMATIC SPAM TAGGING"
    SENDER = "admin@tianyidai.dev"

    RECIPIENT = contents.split("Return-Path: <")[1].split(">")[0]
    EMAIL_RECEIVE_DATE = contents.split("\r\nDate: ")[1].split("\r\n")[0]
    EMAIL_SUBJECT = contents.split("\r\nSubject: ")[1].split("\r\n")[0]
    EMAIL_BODY = body[:240] + (body[240:] and '...')
    CLASSIFICATION = None

    if result["predicted_label"][0][0] == 1.0:
        CLASSIFICATION = "spam"
    else:
        CLASSIFICATION = "not spam"

    CLASSIFICATION_CONFIDENCE_SCORE = result["predicted_probability"][0][0] * 100

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = (
            "We received your email sent at " + EMAIL_RECEIVE_DATE + " with the subject " + EMAIL_SUBJECT + ".\r\n\r\n"
            "Here is a 240 character sample of the email body: " + EMAIL_BODY + "\r\n\r\n"
            "The email was categorized as " + str(
        CLASSIFICATION) + " with a " + str(CLASSIFICATION_CONFIDENCE_SCORE) + "% confidence."
    )
    print(BODY_TEXT)
    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

    return result
