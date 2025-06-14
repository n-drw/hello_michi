import os
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from huggingface_hub import hf_hub_download, list_repo_files

def upload_hf_model_to_s3(repo_id, bucket, s3_prefix):
    s3_client = boto3.client('s3')
    
    # List all files in the repo
    files = list_repo_files(repo_id)
    if not files:
        print(f"No files found for repo_id: {repo_id}")
        sys.exit(1)

    for file in files:
        try:
            # Download or resolve cache path
            local_path = hf_hub_download(repo_id=repo_id, filename=file)
            s3_path = os.path.join(s3_prefix, file).replace('\\', '/')
            print(f"Uploading {local_path} to s3://{bucket}/{s3_path}")
            s3_client.upload_file(local_path, bucket, s3_path)
        except (NoCredentialsError, ClientError, FileNotFoundError) as e:
            print(f"Failed to upload {file}: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python upload_model_to_s3.py <hf_repo_id> <s3_bucket> <s3_prefix>")
        sys.exit(1)

    repo_id = sys.argv[1]
    s3_bucket = sys.argv[2]
    s3_prefix = sys.argv[3]

    upload_hf_model_to_s3(repo_id, s3_bucket, s3_prefix)
    print("Upload complete.")

if __name__ == "__main__":
    main()
