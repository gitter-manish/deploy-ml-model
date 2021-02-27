from boto3.session import Session
import os

ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY']
SECRET_KEY = os.environ['AWS_SECRET_KEY']

def import_data_from_s3():
    session = Session(aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_KEY)

    s3 = session.resource('s3')

    bucket = 'trainwithcnn'

    my_bucket = s3.Bucket(bucket)

    print("Downloading scanned copies...")

    for s3_files in my_bucket.objects.all():
        if s3_files.endswith('.jpg'):
            #my_bucket.download_files(s3_files.key, './'+s3_files.key)
            print(s3_files.key)
            break
        break
