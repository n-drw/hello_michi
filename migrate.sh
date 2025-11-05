#!/bin/bash
# Quick migration script from DeepSeek R1 to Google Gemma
# Run this to upload models and get migration instructions

set -e

echo "=================================="
echo "Gemma Migration Script"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if bucket name is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: S3 bucket name required${NC}"
    echo ""
    echo "Usage: ./migrate.sh YOUR_BUCKET_NAME"
    echo ""
    echo "Example:"
    echo "  ./migrate.sh my-bedrock-models-bucket"
    echo ""
    exit 1
fi

BUCKET_NAME="$1"
AWS_REGION="us-west-2"

echo -e "${YELLOW}Step 1: Verifying AWS credentials...${NC}"
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}✗ AWS credentials not configured${NC}"
    echo "Please run: aws configure"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials verified${NC}"
echo ""

echo -e "${YELLOW}Step 2: Checking S3 bucket...${NC}"
if ! aws s3 ls "s3://${BUCKET_NAME}" > /dev/null 2>&1; then
    echo -e "${YELLOW}Bucket doesn't exist. Creating...${NC}"
    aws s3 mb "s3://${BUCKET_NAME}" --region ${AWS_REGION}
    echo -e "${GREEN}✓ Bucket created${NC}"
else
    echo -e "${GREEN}✓ Bucket exists${NC}"
fi
echo ""

echo -e "${YELLOW}Step 3: Setting up Python virtual environment...${NC}"
# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies in venv
echo "Installing dependencies..."
pip install -q boto3 huggingface_hub

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

echo -e "${YELLOW}Step 4: Updating upload script with bucket name...${NC}"
sed -i.bak "s/your-bedrock-models-bucket/${BUCKET_NAME}/g" upload_gemma_to_s3.py
echo -e "${GREEN}✓ Script updated${NC}"
echo ""

echo -e "${YELLOW}Step 5: Uploading models to S3...${NC}"
echo "This may take 10-30 minutes depending on your connection..."
echo ""

# Ensure we're using venv python
python3 upload_gemma_to_s3.py
echo ""

echo -e "${GREEN}✓ Upload complete!${NC}"
echo ""
echo "=================================="
echo "Next Steps:"
echo "=================================="
echo ""
echo "1. Import model to AWS Bedrock:"
echo "   Go to: https://console.aws.amazon.com/bedrock/imported-models"
echo "   Click: Import model"
echo "   S3 URI: s3://${BUCKET_NAME}/models/gemma-2-2b-it/"
echo ""
echo "2. After import completes, copy the Model ARN"
echo ""
echo "3. Update Rust code:"
echo "   File: /Users/perro/work/michi-api/bedrock_rag/src/conversational_api.rs"
echo "   Replace model ARN in create_teeth_chain() and create_michi_chain()"
echo ""
echo "4. Test the changes:"
echo "   cd /Users/perro/work/michi-api"
echo "   cargo run --example test_teeth"
echo ""
echo "For detailed instructions, see: GEMMA_MIGRATION_GUIDE.md"
echo ""
