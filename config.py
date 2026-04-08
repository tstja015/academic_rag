PAPER_DIRS = [
    "./papers",
    "/home/ans000/doc/CEED",
    "/home/ans000/doc/New_Projects",
    "/home/ans000/doc/MeOH_ADH",
    "/home/ans000/doc/MeOH_FalDH",
    "/home/ans000/doc/MeOH_FDH",
    "/home/ans000/doc/papers",

    # "/path/to/more/papers",
    # "/mnt/nas/research",
]

DB_DIR          = "./db"
COLLECTION_NAME = "academic_papers"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE      = 500

# bedrock settings
BACKEND        = "bedrock"
BEDROCK_REGION = "us-east-2"
BEDROCK_MODEL  = "global.anthropic.claude-sonnet-4-6"

# leave blank -- boto3 reads ~/.aws/credentials automatically
AWS_ACCESS_KEY_ID     = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_SESSION_TOKEN     = ""
