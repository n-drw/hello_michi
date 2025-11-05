#!/usr/bin/env python3
"""
Upload Qwen2.5-0.5B-Instruct (tiny non-reasoning model) to S3 for AWS Bedrock
Much smaller and faster than Gemma 2-2B (~1GB vs ~5GB)
"""
import os
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

def upload_hf_model_to_s3(repo_id, bucket, s3_prefix):
    """Upload model from HuggingFace to S3"""
    s3_client = boto3.client('s3')
    
    print(f"\n{'='*60}")
    print(f"Uploading model: {repo_id}")
    print(f"Target: s3://{bucket}/{s3_prefix}")
    print(f"{'='*60}\n")
    
    # Get list of files
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
    total_size_mb = 0
    
    # Skip unnecessary files
    skip_patterns = ['.md', '.txt', '.gitattributes', 'training_args', '.h5']
    
    for file in files:
        # Skip unwanted files
        if any(pattern in file.lower() for pattern in skip_patterns):
            if file != 'config.json':  # Keep config.json
                print(f"⏭️  Skipping {file}")
                continue
            
        try:
            # Download from HuggingFace cache
            print(f"📥 Downloading {file}...")
            local_path = hf_hub_download(repo_id=repo_id, filename=file)
            
            # Clean S3 path
            s3_path = os.path.join(s3_prefix, file).replace('\\', '/')
            
            # Get file size
            file_size = os.path.getsize(local_path)
            size_mb = file_size / (1024 * 1024)
            total_size_mb += size_mb
            
            print(f"📤 Uploading {file} ({size_mb:.2f} MB)")
            print(f"   S3: s3://{bucket}/{s3_path}")
            
            s3_client.upload_file(local_path, bucket, s3_path)
            uploaded_count += 1
            print(f"   ✅ Success\n")
            
        except (NoCredentialsError, ClientError, FileNotFoundError) as e:
            print(f"   ❌ Failed to upload {file}: {e}\n")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"  ✅ Successful: {uploaded_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📊 Total Size: {total_size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    return failed_count == 0

def main():
    bucket_name = "gemma3-4b-it"  # Your existing bucket
    
    # Configuration for tiny Qwen3 model (with thinking mode control!)
    config = {
        "repo_id": "Qwen/Qwen3-0.6B",
        "bucket": bucket_name,
        "s3_prefix": "models/qwen3-0.6b",
        "description": "Qwen3 0.6B (Tiny LLM with enable_thinking=False)"
    }
    
    print("\n" + "="*60)
    print("AWS Bedrock Tiny Model Upload")
    print("="*60)
    print(f"\n📦 Model: {config['repo_id']}")
    print(f"📏 Size: ~1GB (much smaller than Gemma 2B)")
    print(f"🎯 Use: Fast, non-reasoning chat responses")
    print(f"⚡ Speed: Much faster inference than larger models")
    
    # Verify AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"\n✅ AWS Identity: {identity['Arn']}")
    except Exception as e:
        print(f"\n❌ AWS credentials not configured: {e}")
        print("Please run: aws configure")
        sys.exit(1)
    
    # Upload model
    print(f"\n🚀 Starting upload (estimated time: 5-10 minutes)")
    success = upload_hf_model_to_s3(
        config["repo_id"],
        config["bucket"],
        config["s3_prefix"]
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ UPLOAD COMPLETE!")
        print("="*60)
        print(f"\nS3 Path: s3://{config['bucket']}/{config['s3_prefix']}")
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Import to AWS Bedrock:")
        print("   - Go to: https://console.aws.amazon.com/bedrock/imported-models")
        print("   - Click 'Import model'")
        print(f"   - S3 URI: s3://{config['bucket']}/{config['s3_prefix']}/")
        print("   - Wait ~10-15 minutes for import")
        print("\n2. Copy the Model ARN after import completes")
        print("\n3. Update Rust code:")
        print("   File: /Users/perro/work/michi-api/bedrock_rag/src/conversational_api.rs")
        print('   Replace ARN with your new Qwen model ARN')
        print("\n4. Configure for albinoteeth:")
        print("   .with_max_tokens(512)")
        print("   .with_temperature(0.7)  // Non-thinking mode params")
        print("   .with_top_p(0.8)        // Non-thinking mode params")
        print("   .with_strip_reasoning(true)  // Extra safety")
        print("\n5. Test:")
        print("   cd /Users/perro/work/michi-api")
        print("   cargo run --example test_teeth")
        print("\n" + "="*60)
        print("\n💡 Benefits of Qwen3-0.6B:")
        print("  - Much faster upload (~1.2GB vs ~5GB)")
        print("  - Much faster inference (0.6B params)")
        print("  - Lower cost")
        print("  - Built-in enable_thinking=False mode!")
        print("  - No reasoning output (direct answers!)")
        print("  - Latest Qwen3 architecture (May 2025)")
        print("  - Perfect for character chat")
        print("="*60 + "\n")
    else:
        print("\n❌ Upload failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
