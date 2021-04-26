import json
import os
import boto3
from utils import *
from io import StringIO
sagemaker = boto3.client('runtime.sagemaker')
ENDPOINT_NAME = 'HW3-SpamDetector-API-Endpoint'
vocabulary_length = 9013


test = ['MN2PR18MB287785CE575790028AF73E7BA6759MN2PR18MB2877namp_\r\nContent-Type: text/plain; charset="us-ascii"\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n\r\nWhat is Lorem Ipsum?\r\nLorem Ipsum is simply dummy text of the printing and typesetting industry. =\r\nLorem Ipsum has been the industry\'s standard dummy text ever since the 1500=\r\ns, when an unknown printer took a galley of type and scrambled it to make a=\r\n type specimen book. It has survived not only five centuries, but also the =\r\nleap into electronic typesetting, remaining essentially unchanged. It was p=\r\nopularised in the 1960s with the release of Letraset sheets containing Lore=\r\nm Ipsum passages, and more recently with desktop publishing software like A=\r\nldus PageMaker including versions of Lorem Ipsum.\r\n\r\nBlah blah\r\n\r\n\r\n\r\nWhat is Lorem Ipsum?\r\nLorem Ipsum is simply dummy text of the printing and typesetting industry. =\r\nLorem Ipsum has been the industry\'s standard dummy text ever since the 1500=\r\ns, when an unknown printer took a galley of type and scrambled it to make a=\r\n type specimen book. It has survived not only five centuries, but also the =\r\nleap into electronic typesetting, remaining essentially unchanged. It was p=\r\nopularised in the 1960s with the release of Letraset sheets containing Lore=\r\nm Ipsum passages, and more recently with desktop publishing software like A=\r\nldus PageMaker including versions of Lorem Ipsum.\r\n\r\nBlah blah\r\n\r\n\r\n']
body ="".join(test[0].split('\r\n\r\n')[1:])
body = "".join(body.splitlines()).replace("=","")
# print([body])


test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
# test_messages = [body]
one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
# print(one_hot_test_messages)
encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
# print(encoded_test_messages)

io = StringIO()
json.dump(encoded_test_messages.tolist(), io)
response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                     ContentType='application/python-pickle',
                                     Body = bytes(io.getvalue(),'utf-8')  )
# print(response)
result = response['Body'].read()
print(result)

result = json.loads(result)

f = open("t3.eml",'r')
contents=f.read()
# print(contents)
#### Return Msg

SUBJECT = "AUTOMATIC SPAM TAGGING"
SENDER = "admin@tianyidai.dev"

##### Notice here parsing is different from AWS \r\n
RECIPIENT = contents.split("Return-Path: <")[1].split(">")[0]
EMAIL_RECEIVE_DATE = contents.split("\nDate: ")[1].split("\n")[0]
# print(EMAIL_RECEIVE_DATE)
EMAIL_SUBJECT = contents.split("\nSubject: ")[1].split("\n")[0]
EMAIL_BODY = "".join( contents.split("--_000_")[1].split("\n\n")[1:])
EMAIL_BODY = " ".join(EMAIL_BODY.splitlines()).replace("=","")
EMAIL_BODY = EMAIL_BODY[:240] + (EMAIL_BODY[240:] and '...')
# print(EMAIL_BODY)
CLASSIFICATION = None
if result["predicted_label"][0][0] == 1.0:
    CLASSIFICATION = "spam"
else:
    CLASSIFICATION = "not spam"

CLASSIFICATION_CONFIDENCE_SCORE = result["predicted_probability"][0][0] * 100

# The email body for recipients with non-HTML email clients.
BODY_TEXT = (
        "We received your email sent at " + EMAIL_RECEIVE_DATE + " with the subject " + EMAIL_SUBJECT + ".\r\n\r\n"
                                                                                                        "Here is "
                                                                                                        "a sample of the email body: " + EMAIL_BODY + "\r\n\r\n"
                                                                                                                                                                            "The email was categorized as " + str(
    CLASSIFICATION) + " with a " + str(CLASSIFICATION_CONFIDENCE_SCORE) + "% confidence."
)
print(BODY_TEXT)
