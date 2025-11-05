#!/usr/bin/env python3
"""
Upload Google Gemma and Stella embedding model to S3 for AWS Bedrock import.
This script downloads models from HuggingFace and uploads them to S3.
"""
import os
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

def upload_hf_model_to_s3(repo_id, bucket, s3_prefix):
    """
    Upload all files from a HuggingFace model to S3.
    
    Args:
        repo_id: HuggingFace model ID (e.g., "google/gemma-2-2b-it")
        bucket: S3 bucket name
        s3_prefix: Prefix path in S3 (e.g., "models/gemma-2-2b")
    """
    s3_client = boto3.client('s3')
    
    print(f"\n{'='*60}")
    print(f"Uploading model: {repo_id}")
    print(f"Target: s3://{bucket}/{s3_prefix}")
    print(f"{'='*60}\n")
    
    # List all files in the repo
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"❌ Failed to list files for {repo_id}: {e}")
        return False
    
    if not files:
        print(f"❌ No files found for repo_id: {repo_id}")
        return False

    uploaded_count = 0
    failed_count = 0
    
    for file in files:
        # Skip unnecessary files
        if file.endswith(('.md', '.txt', '.gitattributes')) and file != 'config.json':
            print(f"⏭️  Skipping {file}")
            continue
            
        try:
            # Download from HuggingFace cache
            local_path = hf_hub_download(repo_id=repo_id, filename=file)
            
            # Clean S3 path
            s3_path = os.path.join(s3_prefix, file).replace('\\', '/')
            
            # Get file size
            file_size = os.path.getsize(local_path)
            size_mb = file_size / (1024 * 1024)
            
            print(f"📤 Uploading {file} ({size_mb:.2f} MB)")
            print(f"   Local: {local_path}")
            print(f"   S3: s3://{bucket}/{s3_path}")
            
            s3_client.upload_file(local_path, bucket, s3_path)
            uploaded_count += 1
            print(f"   ✅ Success\n")
            
        except (NoCredentialsError, ClientError, FileNotFoundError) as e:
            print(f"   ❌ Failed to upload {file}: {e}\n")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Upload Summary for {repo_id}:")
    print(f"  ✅ Successful: {uploaded_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"{'='*60}\n")
    
    return failed_count == 0

def main():
    # Configuration for models
    configs = [
        {
            "repo_id": "google/gemma-2-2b-it",
            "bucket": "gemma3-4b-it",  # TODO: Replace with your bucket
            "s3_prefix": "models/gemma-2-2b-it",
            "description": "Google Gemma 2B Instruct (Non-reasoning LLM)"
        },
        {
            "repo_id": "NovaSearch/stella_en_400M_v5",
            "bucket": "gemma3-4b-it",  # TODO: Replace with your bucket
            "s3_prefix": "models/stella-en-400m-v5",
            "description": "Stella 400M Embedding Model"
        }
    ]
    
    # Allow override via command line
    if len(sys.argv) >= 3:
        configs = [{
            "repo_id": sys.argv[1],
            "bucket": sys.argv[2],
            "s3_prefix": sys.argv[3] if len(sys.argv) >= 4 else f"models/{sys.argv[1].split('/')[-1]}",
            "description": "Custom model"
        }]
    
    print("\n" + "="*60)
    print("AWS Bedrock Model Upload Script")
    print("="*60)
    
    # Verify AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"\n✅ AWS Identity: {identity['Arn']}")
    except Exception as e:
        print(f"\n❌ AWS credentials not configured: {e}")
        print("Please run: aws configure")
        sys.exit(1)
    
    results = []
    for config in configs:
        print(f"\n📦 Processing: {config['description']}")
        print(f"   Repo: {config['repo_id']}")
        
        success = upload_hf_model_to_s3(
            config["repo_id"],
            config["bucket"],
            config["s3_prefix"]
        )
        
        results.append({
            "model": config["repo_id"],
            "success": success,
            "s3_path": f"s3://{config['bucket']}/{config['s3_prefix']}"
        })
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{status} - {result['model']}")
        print(f"  S3 Path: {result['s3_path']}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Import models to AWS Bedrock:")
    print("   - Go to AWS Bedrock Console → Imported Models")
    print("   - Click 'Import model'")
    print("   - Select S3 paths above")
    print("   - Note the model ARN after import")
    print("\n2. Update your Rust code with new model ARN")
    print("\n3. Test with the new non-reasoning model!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
