"""
setup_aws.py — Unified AWS Infrastructure & Data Setup
=======================================================
Manages all AWS resources for the Card Fraud Detection MLOps project.

Commands:
  python setup_aws.py create       →  Create S3 buckets + ECR repo + RDS instance + populate DB
  python setup_aws.py create-s3    →  Create S3 buckets only
  python setup_aws.py create-ecr   →  Create ECR repository only
  python setup_aws.py create-rds   →  Create RDS instance + IAM role only
  python setup_aws.py populate     →  Populate RDS from S3 (raw CSVs → feature tables)
  python setup_aws.py upload       →  Upload feather/pkl ML files to S3
  python setup_aws.py delete       →  Destroy ALL resources (S3 + ECR + RDS + IAM)
  python setup_aws.py delete-s3    →  Destroy S3 buckets only
  python setup_aws.py delete-rds   →  Destroy RDS instance + IAM role only
  python setup_aws.py help         →  Show this message

Resources managed:
  • s3://sagemaker-datasets-{account_id}       ← feather/pkl ML files
  • s3://mlops-models-{account_id}             ← model artifacts
  • s3://card-fraud-{account_id}-{region}      ← raw CSV datasets (for RDS import)
  • ecr://{account_id}.../mlops-pipeline-pretrain
  • ecr://{account_id}.../mlops-pipeline-train
  • rds://fraud-detection-db (PostgreSQL)
  • iam://fraud-rds-s3-role
"""

import argparse
import json
import logging
import os
import sys
import time

import boto3
import pandas as pd
import psycopg2
from botocore.exceptions import ClientError
from psycopg2 import sql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION     = "us-east-1"
RDS_PORT   = 5433

# All bucket/resource names are derived from the account id — never hardcoded.
def _get_account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]

def _bucket_sagemaker(account_id: str) -> str:
    return f"sagemaker-datasets-{account_id}"

def _bucket_mlops(account_id: str) -> str:
    return f"mlops-models-{account_id}"

def _bucket_raw_data(account_id: str, region: str) -> str:
    """Bucket used by your friend's RDS populate flow (raw CSV storage)."""
    return f"card-fraud-{account_id}-{region}"

# ECR repo names — two repos, one per Docker image (pretrain CPU / train GPU)
ECR_PRETRAIN_REPO = "mlops-pipeline-pretrain"
ECR_TRAIN_REPO    = "mlops-pipeline-train"

# RDS / IAM
DB_INSTANCE_ID    = "fraud-detection-db"
DB_INSTANCE_CLASS = "db.t3.micro"
DB_MASTER_USER    = "postgres"
DB_MASTER_PASS    = "changeme"          # override via env var DB_MASTER_PASSWORD
IAM_ROLE_NAME     = "fraud-rds-s3-role"

# ML files to upload (feather + pkl → sagemaker / mlops buckets)
def _ml_upload_manifest(account_id: str) -> list[dict]:
    data_bucket  = _bucket_sagemaker(account_id)
    model_bucket = _bucket_mlops(account_id)
    return [
        {
            "local":  "data_ieee/transactions_train.feather",
            "bucket": data_bucket,
            "key":    "data/transactions_train.feather",
        },
        {
            "local":  "data_ieee/transactions_test.feather",
            "bucket": data_bucket,
            "key":    "data/transactions_test.feather",
        },
        {
            "local":  "models/preprocessing_config.pkl",
            "bucket": model_bucket,
            "key":    "models/preprocessing_config.pkl",
        },
    ]

# Raw CSV files for RDS population (your friend's flow)
RAW_CSV_FILES = [
    "train_identity.csv",
    "train_transaction.csv",
    "test_identity.csv",
    "test_transaction.csv",
]
RAW_DATASETS_DIR = os.path.join(os.path.dirname(__file__), "data_ieee", "ieee-fraud-detection")

# Retry / wait constants (RDS s3Import feature needs time to propagate)
IAM_PROPAGATION_WAIT       = 15   # seconds
S3_IMPORT_PROPAGATION_WAIT = 30   # seconds
S3_IMPORT_MAX_RETRIES      = 5
S3_IMPORT_RETRY_WAIT       = 15   # seconds

# DB schema constants
DTYPES_MAP  = {"string": "VARCHAR", "Int64": "BIGINT", "Float64": "DOUBLE PRECISION"}
PRIMARY_KEY = "transactionid"
TABLE_NAMES = ["identity", "transaction"]
SPLITS      = ["train", "test"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def build_session() -> boto3.Session:
    return boto3.Session(region_name=REGION)

def build_clients(session: boto3.Session) -> dict:
    return {
        "s3":  session.client("s3",  region_name=REGION),
        "sts": session.client("sts", region_name=REGION),
        "iam": session.client("iam", region_name=REGION),
        "rds": session.client("rds", region_name=REGION),
        "ecr": session.client("ecr", region_name=REGION),
        "ec2": session.client("ec2", region_name=REGION),
    }

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _create_bucket(s3_client, bucket_name: str) -> None:
    try:
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        s3_client.get_waiter("bucket_exists").wait(Bucket=bucket_name)
        logger.info(f"[S3] ✅ Created  →  s3://{bucket_name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info(f"[S3] ⚠️  Already exists  →  s3://{bucket_name}  (skipping)")
        else:
            logger.error(f"[S3] ❌ Failed to create {bucket_name}: {e}")
            raise


def _delete_bucket(s3_client, bucket_name: str) -> None:
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name):
            objects = page.get("Contents", [])
            if objects:
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
                logger.info(f"[S3] Deleted {len(objects)} object(s) from {bucket_name}")
        s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"[S3] 🗑️  Deleted  →  s3://{bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            logger.info(f"[S3] ⚠️  Already gone  →  s3://{bucket_name}  (skipping)")
        else:
            logger.error(f"[S3] ❌ Failed to delete {bucket_name}: {e}")
            raise


def _list_bucket_contents(s3_client, bucket_name: str) -> list[str] | None:
    objects = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                objects.append(obj["Key"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            logger.info(f"[S3] ⚠️  Bucket does not exist  →  s3://{bucket_name}")
            return None
        raise

    if objects:
        logger.info(f"[S3] 📂 s3://{bucket_name}  ({len(objects)} object(s)):")
        for key in objects:
            logger.info(f"[S3]     • {key}")
    else:
        logger.info(f"[S3] 📂 s3://{bucket_name}  (empty)")
    return objects


def create_s3(clients: dict, account_id: str) -> None:
    """Create all three S3 buckets."""
    logger.info(f"\n[S3] Creating buckets in [{REGION}] for account [{account_id}]")
    for bucket_name in [
        _bucket_sagemaker(account_id),
        _bucket_mlops(account_id),
        _bucket_raw_data(account_id, REGION),
    ]:
        _create_bucket(clients["s3"], bucket_name)

    logger.info("[S3] All buckets ready. Add these ARNs to your IAM policy:")
    for bucket_name in [_bucket_sagemaker(account_id), _bucket_mlops(account_id)]:
        logger.info(f'  "arn:aws:s3:::{bucket_name}"')
        logger.info(f'  "arn:aws:s3:::{bucket_name}/*"')


def delete_s3(clients: dict, account_id: str) -> None:
    """Empty and delete all three S3 buckets."""
    logger.info(f"\n[S3] Scanning buckets before deletion …")
    for bucket_name in [
        _bucket_sagemaker(account_id),
        _bucket_mlops(account_id),
        _bucket_raw_data(account_id, REGION),
    ]:
        _list_bucket_contents(clients["s3"], bucket_name)

    logger.info("[S3] Deleting all buckets …")
    for bucket_name in [
        _bucket_sagemaker(account_id),
        _bucket_mlops(account_id),
        _bucket_raw_data(account_id, REGION),
    ]:
        _delete_bucket(clients["s3"], bucket_name)


def _s3_progress_callback(filename: str, file_size: int):
    """Return a callback that logs upload progress every 10%."""
    uploaded  = [0]
    last_pct  = [-1]

    def callback(bytes_transferred: int):
        uploaded[0] += bytes_transferred
        pct       = int(uploaded[0] / file_size * 100)
        milestone = (pct // 10) * 10
        if milestone > last_pct[0]:
            last_pct[0] = milestone
            mb_done  = uploaded[0] / 1_048_576
            mb_total = file_size   / 1_048_576
            logger.info(f"[S3]   {filename}: {mb_done:.1f} / {mb_total:.1f} MB  ({milestone}%)")

    return callback


def upload_raw_csvs(clients: dict, account_id: str) -> None:
    """Upload raw IEEE fraud CSVs to the card-fraud bucket (feeds RDS populate)."""
    bucket_name = _bucket_raw_data(account_id, REGION)
    logger.info(f"[S3] Uploading raw CSVs to s3://{bucket_name}/raw/")
    for filename in RAW_CSV_FILES:
        local_path = os.path.join(RAW_DATASETS_DIR, filename)
        s3_key     = f"raw/{filename}"
        if not os.path.exists(local_path):
            logger.error(f"[S3] File not found: {local_path}")
            continue
        file_size = os.path.getsize(local_path)
        logger.info(f"[S3] Uploading {filename} ({file_size / 1_048_576:.1f} MB) ...")
        clients["s3"].upload_file(
            local_path, bucket_name, s3_key,
            Callback=_s3_progress_callback(filename, file_size),
        )
        logger.info(f"[S3] Done: s3://{bucket_name}/{s3_key}")
    logger.info("[S3] Raw CSV upload complete")


def upload_ml_files(clients: dict, account_id: str) -> None:
    """Upload processed feather + pkl ML files to the sagemaker/mlops buckets."""
    logger.info("[S3] Uploading ML files (feather + pkl) …")
    manifest = _ml_upload_manifest(account_id)
    for item in manifest:
        if not os.path.exists(item["local"]):
            logger.error(f"[S3] ❌ Local file not found: {item['local']}")
            continue
        logger.info(f"[S3] Uploading {item['local']} → s3://{item['bucket']}/{item['key']}")
        clients["s3"].upload_file(item["local"], item["bucket"], item["key"])
        logger.info(f"[S3] ✅ Uploaded to s3://{item['bucket']}/{item['key']}")
    logger.info("[S3] ML file upload complete")

# ---------------------------------------------------------------------------
# ECR helpers
# ---------------------------------------------------------------------------

def _create_ecr_repo(ecr_client, repo_name: str, account_id: str) -> None:
    repo_uri = f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{repo_name}"
    try:
        ecr_client.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="MUTABLE",
        )
        logger.info(f"[ECR] ✅ Created  →  {repo_uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            logger.info(f"[ECR] ⚠️  Already exists  →  {repo_uri}  (skipping)")
        else:
            logger.error(f"[ECR] ❌ Failed to create {repo_name}: {e}")
            raise


def _delete_ecr_repo(ecr_client, repo_name: str, account_id: str) -> None:
    repo_uri = f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{repo_name}"
    try:
        ecr_client.delete_repository(repositoryName=repo_name, force=True)
        logger.info(f"[ECR] 🗑️  Deleted  →  {repo_uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            logger.info(f"[ECR] ⚠️  Already gone  →  {repo_uri}  (skipping)")
        else:
            logger.error(f"[ECR] ❌ Failed to delete {repo_name}: {e}")
            raise


def create_ecr(clients: dict, account_id: str) -> None:
    """Create both ECR repositories (pretrain + train)."""
    logger.info(f"\n[ECR] Creating repositories in [{REGION}] for account [{account_id}]")
    _create_ecr_repo(clients["ecr"], ECR_PRETRAIN_REPO, account_id)
    _create_ecr_repo(clients["ecr"], ECR_TRAIN_REPO, account_id)
    logger.info("[ECR] Both repositories ready.")
    logger.info("Build and push your images with:  make all")


def delete_ecr(clients: dict, account_id: str) -> None:
    """Delete both ECR repositories."""
    logger.info("\n[ECR] Deleting repositories …")
    _delete_ecr_repo(clients["ecr"], ECR_PRETRAIN_REPO, account_id)
    _delete_ecr_repo(clients["ecr"], ECR_TRAIN_REPO, account_id)

# ---------------------------------------------------------------------------
# IAM helpers
# ---------------------------------------------------------------------------

def _assume_role_policy_doc() -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect":    "Allow",
            "Principal": {"Service": "rds.amazonaws.com"},
            "Action":    "sts:AssumeRole",
        }],
    })


def _s3_read_policy_doc(bucket_name: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect":   "Allow",
            "Action":   ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
        }],
    })


def _s3_write_policy_doc(bucket_name: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect":   "Allow",
                "Action":   ["s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            },
            {
                "Effect":   "Allow",
                "Action":   ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket_name}"],
            },
        ],
    })


def _ensure_iam_role(iam_client, account_id: str) -> str:
    """Create (or retrieve) the IAM role for RDS → S3 access. Returns the role ARN."""
    bucket_name = _bucket_raw_data(account_id, REGION)

    logger.info(f"[IAM] Creating role: {IAM_ROLE_NAME}")
    try:
        role = iam_client.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=_assume_role_policy_doc(),
        )
        role_arn = role["Role"]["Arn"]
        logger.info(f"[IAM] ✅ Role created: {role_arn}")
    except iam_client.exceptions.EntityAlreadyExistsException:
        role_arn = iam_client.get_role(RoleName=IAM_ROLE_NAME)["Role"]["Arn"]
        logger.info(f"[IAM] ⚠️  Role already exists: {role_arn}")

    iam_client.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName="rds-s3-read-policy",
        PolicyDocument=_s3_read_policy_doc(bucket_name),
    )
    iam_client.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName="rds-s3-write-policy",
        PolicyDocument=_s3_write_policy_doc(bucket_name),
    )
    logger.info(f"[IAM] Waiting {IAM_PROPAGATION_WAIT}s for IAM propagation …")
    time.sleep(IAM_PROPAGATION_WAIT)
    return role_arn


def _delete_iam_role(iam_client) -> None:
    """Detach inline policies and delete the IAM role."""
    for policy_name in ["rds-s3-read-policy", "rds-s3-write-policy"]:
        try:
            iam_client.delete_role_policy(RoleName=IAM_ROLE_NAME, PolicyName=policy_name)
            logger.info(f"[IAM] Deleted policy: {policy_name}")
        except iam_client.exceptions.NoSuchEntityException:
            logger.info(f"[IAM] Policy {policy_name} not found, skipping")
    try:
        iam_client.delete_role(RoleName=IAM_ROLE_NAME)
        logger.info(f"[IAM] 🗑️  Deleted role: {IAM_ROLE_NAME}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"[IAM] Role {IAM_ROLE_NAME} not found, skipping")

# ---------------------------------------------------------------------------
# RDS helpers
# ---------------------------------------------------------------------------

def _authorize_rds_ingress(clients: dict, db_instance_identifier: str, port: int) -> None:
    """Authorize ingress on the RDS instance's VPC security group for the specified port."""
    try:
        instance = clients["rds"].describe_db_instances(
            DBInstanceIdentifier=db_instance_identifier
        )["DBInstances"][0]
        sg_id = instance["VpcSecurityGroups"][0]["VpcSecurityGroupId"]
        
        try:
            clients["ec2"].authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    }
                ],
            )
            logger.info(f"[RDS] Authorized public ingress to security group {sg_id} on port {port}.")
        except clients["ec2"].exceptions.ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                logger.info(f"[RDS] Security group rule already exists for port {port} on {sg_id}.")
            else:
                logger.error(f"[RDS] Failed to authorize ingress: {e}")
                raise
    except Exception as e:
        logger.warning(f"[RDS] Could not authorize SG ingress automatically: {e}")


def create_rds(clients: dict, account_id: str) -> None:
    """Create IAM role + RDS instance and link them."""
    db_password = os.getenv("DB_MASTER_PASSWORD", DB_MASTER_PASS)
    role_arn = _ensure_iam_role(clients["iam"], account_id)

    logger.info(f"[RDS] Creating DB instance: {DB_INSTANCE_ID}")
    try:
        clients["rds"].create_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_ID,
            AllocatedStorage=20,
            DBInstanceClass=DB_INSTANCE_CLASS,
            Engine="postgres",
            MasterUsername=DB_MASTER_USER,
            MasterUserPassword=db_password,
            PubliclyAccessible=True,
            Port=RDS_PORT,
        )
    except clients["rds"].exceptions.DBInstanceAlreadyExistsFault:
        logger.info("[RDS] ⚠️  DB instance already exists, skipping creation")

    logger.info("[RDS] Waiting for instance to become available …")
    clients["rds"].get_waiter("db_instance_available").wait(
        DBInstanceIdentifier=DB_INSTANCE_ID
    )
    logger.info("[RDS] ✅ Instance is available")

    logger.info("[RDS] Associating IAM role for s3Import …")
    try:
        clients["rds"].add_role_to_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_ID,
            RoleArn=role_arn,
            FeatureName="s3Import",
        )
        clients["rds"].get_waiter("db_instance_available").wait(
            DBInstanceIdentifier=DB_INSTANCE_ID
        )
        logger.info(f"[RDS] Waiting {S3_IMPORT_PROPAGATION_WAIT}s for s3Import propagation …")
        time.sleep(S3_IMPORT_PROPAGATION_WAIT)
    except Exception as e:
        msg = str(e)
        if "is already associated" in msg or "supports only one ARN" in msg:
            logger.info("[RDS] s3Import role already associated, skipping")
        else:
            raise

    instance = clients["rds"].describe_db_instances(
        DBInstanceIdentifier=DB_INSTANCE_ID
    )["DBInstances"][0]
    host = instance["Endpoint"]["Address"]
    port = instance["Endpoint"]["Port"]
    logger.info(f"[RDS] ✅ Endpoint: {host}:{port}")

    # Ensure security group rule allows inbound connection so script doesn't time out
    _authorize_rds_ingress(clients, DB_INSTANCE_ID, RDS_PORT)


def delete_rds(clients: dict) -> None:
    """Delete the RDS instance and IAM role."""
    logger.info(f"[RDS] Deleting DB instance: {DB_INSTANCE_ID}")
    try:
        clients["rds"].delete_db_instance(
            DBInstanceIdentifier=DB_INSTANCE_ID,
            SkipFinalSnapshot=True,
            DeleteAutomatedBackups=True,
        )
        logger.info("[RDS] Waiting for instance to be fully removed …")
        clients["rds"].get_waiter("db_instance_deleted").wait(
            DBInstanceIdentifier=DB_INSTANCE_ID
        )
        logger.info("[RDS] 🗑️  Instance deleted")
    except clients["rds"].exceptions.DBInstanceNotFoundFault:
        logger.info("[RDS] Instance not found, nothing to delete")

    _delete_iam_role(clients["iam"])

# ---------------------------------------------------------------------------
# DB Population helpers (your friend's flow)
# ---------------------------------------------------------------------------

def _get_rds_connection(clients: dict):
    """Return an open psycopg2 connection to the RDS instance."""
    db_password = os.getenv("DB_MASTER_PASSWORD", DB_MASTER_PASS)
    instance = clients["rds"].describe_db_instances(
        DBInstanceIdentifier=DB_INSTANCE_ID
    )["DBInstances"][0]
    host = instance["Endpoint"]["Address"]
    port = instance["Endpoint"]["Port"]
    return psycopg2.connect(
        host=host, port=port,
        database="postgres",
        user=DB_MASTER_USER,
        password=db_password,
        connect_timeout=10,
    )


def _infer_columns_from_csv() -> dict:
    """Read the first 100 rows of each CSV to infer column names and SQL types."""
    columns = {}
    for split in SPLITS:
        columns[split] = {}
        for table_name in TABLE_NAMES:
            filepath = os.path.join(RAW_DATASETS_DIR, f"{split}_{table_name}.csv")
            chunk = pd.read_csv(filepath, nrows=100, dtype_backend="numpy_nullable")
            col_defs = []
            for col, dtype in chunk.dtypes.items():
                sql_type = DTYPES_MAP.get(str(dtype))
                if sql_type is None:
                    raise ValueError(
                        f'Unknown dtype "{dtype}" for column "{col}" in '
                        f"{split}_{table_name}. Add it to DTYPES_MAP."
                    )
                col_defs.append((col.lower().replace("-", "_"), sql_type))
            columns[split][table_name] = col_defs
    return columns


def _create_raw_schema(cur, columns: dict) -> None:
    """Create raw schema + identity/transaction tables for each split."""
    logger.info("[DB] Creating raw schema and tables …")
    cur.execute("CREATE EXTENSION IF NOT EXISTS aws_commons;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS aws_s3;")
    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier("raw")))

    for table_name in TABLE_NAMES:
        col_defs = columns["train"][table_name]   # train = superset of columns
        col_sql = sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(sql_type))
            for col_name, sql_type in col_defs
        )
        for split in SPLITS:
            full_table = f"{split}_{table_name}"
            logger.info(f"[DB]   Creating table raw.{full_table}")
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;").format(
                sql.Identifier("raw"), sql.Identifier(full_table)
            ))
            cur.execute(sql.SQL("""
                CREATE TABLE {}.{} (
                    {},
                    PRIMARY KEY ({})
                );
            """).format(
                sql.Identifier("raw"),
                sql.Identifier(full_table),
                col_sql,
                sql.Identifier(PRIMARY_KEY),
            ))
    logger.info("[DB] Raw schema created")


def _import_from_s3(cur, conn, columns: dict, account_id: str) -> None:
    """Import each CSV from the card-fraud S3 bucket into the raw tables."""
    bucket_name = _bucket_raw_data(account_id, REGION)
    logger.info("[DB] Importing data from S3 …")

    for table_name in TABLE_NAMES:
        for split in SPLITS:
            col_defs = columns[split][table_name]
            column_string = ",".join(col_name for col_name, _ in col_defs)
            full_table = f"raw.{split}_{table_name}"
            s3_key = f"raw/{split}_{table_name}.csv"

            import_query = sql.SQL("""
                SELECT aws_s3.table_import_from_s3(
                    {},
                    {},
                    '(FORMAT csv, HEADER true)',
                    aws_commons.create_s3_uri({}, {}, {})
                );
            """).format(
                sql.Literal(full_table),
                sql.Literal(column_string),
                sql.Literal(bucket_name),
                sql.Literal(s3_key),
                sql.Literal(REGION),
            )

            for attempt in range(1, S3_IMPORT_MAX_RETRIES + 1):
                try:
                    logger.info(f"[DB]   Importing s3://{bucket_name}/{s3_key} → {full_table}")
                    cur.execute(import_query)
                    conn.commit()
                    logger.info(f"[DB]   ✅ {full_table} imported")
                    break
                except psycopg2.errors.InternalError_ as e:
                    conn.rollback()
                    if "s3Import" in str(e) and attempt < S3_IMPORT_MAX_RETRIES:
                        logger.info(
                            f"[DB]   ⏳ s3Import not ready — retry {attempt}/{S3_IMPORT_MAX_RETRIES} "
                            f"in {S3_IMPORT_RETRY_WAIT}s …"
                        )
                        time.sleep(S3_IMPORT_RETRY_WAIT)
                    else:
                        raise

    logger.info("[DB] S3 import complete")


def _create_feature_schema(cur) -> None:
    """Join transaction + identity into the feature schema (one table per split)."""
    logger.info("[DB] Creating feature schema …")

    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'train_identity' AND table_schema = 'raw'
    """)
    identity_columns = [c for (c,) in cur.fetchall()]

    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier("feature")))

    # Rename identity columns with _new suffix to avoid name collisions on the join
    columns_query = sql.SQL(", ").join(
        sql.SQL("i.{} AS {}").format(
            sql.Identifier(col),
            sql.Identifier(col + "_new"),
        )
        for col in identity_columns
    )

    for split in SPLITS:
        logger.info(f"[DB]   Creating feature.{split} (transaction LEFT JOIN identity)")
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;").format(
            sql.Identifier("feature"), sql.Identifier(split)
        ))
        cur.execute(sql.SQL("""
            CREATE TABLE {}.{} AS
            SELECT t.*, {}
            FROM raw.{} AS t
            LEFT JOIN raw.{} AS i
                ON t.transactionid = i.transactionid
        """).format(
            sql.Identifier("feature"),
            sql.Identifier(split),
            columns_query,
            sql.Identifier(split + "_transaction"),
            sql.Identifier(split + "_identity"),
        ))

    logger.info("[DB] Feature schema created")


def populate_rds(clients: dict, account_id: str) -> None:
    """
    DB population flow:
      1. Upload raw CSVs to the card-fraud S3 bucket
      2. Create raw schema + tables
      3. Import CSVs from S3 → raw tables

    The feature JOIN (transaction LEFT JOIN identity) is now done at query time
    inside steps/data_split.py, so no feature schema is created here.
    """
    # Upload CSVs first so the RDS s3Import has something to read
    upload_raw_csvs(clients, account_id)

    columns = _infer_columns_from_csv()

    logger.info("[DB] Connecting to RDS …")
    conn = _get_rds_connection(clients)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        _create_raw_schema(cur, columns)
        conn.commit()

        _import_from_s3(cur, conn, columns, account_id)

        # Verify row counts in raw tables
        for split in SPLITS:
            for table_name in TABLE_NAMES:
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}.{};").format(
                    sql.Identifier("raw"),
                    sql.Identifier(f"{split}_{table_name}"),
                ))
                count = cur.fetchone()[0]
                logger.info(f"[DB] raw.{split}_{table_name}: {count:,} rows")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("[DB] Connection closed")

# ---------------------------------------------------------------------------
# Composite commands
# ---------------------------------------------------------------------------

def cmd_create(clients: dict, account_id: str) -> None:
    """Full create: S3 + ECR + RDS + populate DB + upload ML files."""
    create_s3(clients, account_id)
    create_ecr(clients, account_id)
    create_rds(clients, account_id)
    populate_rds(clients, account_id)
    upload_ml_files(clients, account_id)


def cmd_delete(clients: dict, account_id: str) -> None:
    """Full teardown with confirmation: S3 + ECR + RDS + IAM."""
    logger.info(f"\n🔍 Scanning all resources for account [{account_id}] …\n")
    _list_bucket_contents(clients["s3"], _bucket_sagemaker(account_id))
    _list_bucket_contents(clients["s3"], _bucket_mlops(account_id))
    _list_bucket_contents(clients["s3"], _bucket_raw_data(account_id, REGION))
    logger.info(f"[ECR] {ECR_PRETRAIN_REPO}  &  {ECR_TRAIN_REPO}")
    logger.info(f"[RDS] {DB_INSTANCE_ID}")

    print("\n" + "─" * 60)
    print("⚠️  WARNING: This permanently deletes ALL resources above.")
    print("             S3 files, Docker images, and the RDS database.")
    print("─" * 60)
    confirm = input("\nType  YES  to confirm deletion: ").strip()

    if confirm != "YES":
        logger.info("Cancelled. Nothing was deleted.")
        return

    delete_rds(clients)
    delete_s3(clients, account_id)
    delete_ecr(clients, account_id)
    logger.info("\n✅ All resources deleted.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = {
    "create":     "Create all resources (S3 + ECR + RDS + populate + upload ML files)",
    "create-s3":  "Create S3 buckets only",
    "create-ecr": "Create ECR repositories only",
    "create-rds": "Create RDS instance + IAM role only",
    "populate":   "Populate RDS from raw CSVs (uploads CSVs → imports → builds feature tables)",
    "upload":     "Upload feather + pkl ML files to S3",
    "delete":     "Destroy ALL resources (S3 + ECR + RDS + IAM) — asks for confirmation",
    "delete-s3":  "Destroy S3 buckets only",
    "delete-rds": "Destroy RDS instance + IAM role only",
    "help":       "Show this help message",
}


def print_help(account_id: str | None = None) -> None:
    acct = account_id or "<account_id>"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AWS MLOps Setup — Card Fraud Detection             ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python setup_aws.py <command>

Commands:
""")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<14}  {desc}")
    print(f"""
Resources managed:
  • s3://sagemaker-datasets-{acct}
  • s3://mlops-models-{acct}
  • s3://card-fraud-{acct}-{REGION}
  • ecr://{acct}.dkr.ecr.{REGION}.amazonaws.com/{ECR_PRETRAIN_REPO}
  • ecr://{acct}.dkr.ecr.{REGION}.amazonaws.com/{ECR_TRAIN_REPO}
  • rds://{DB_INSTANCE_ID} (PostgreSQL)
  • iam://{IAM_ROLE_NAME}

Environment variable:
  DB_MASTER_PASSWORD  Override the default RDS master password (recommended)
""")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"\n  Usage: python setup_aws.py [{' | '.join(COMMANDS)}]\n")
        sys.exit(1)

    command = sys.argv[1]

    if command == "help":
        print_help()
        return

    session    = build_session()
    clients    = build_clients(session)
    account_id = _get_account_id(session)

    dispatch = {
        "create":     lambda: cmd_create(clients, account_id),
        "create-s3":  lambda: create_s3(clients, account_id),
        "create-ecr": lambda: create_ecr(clients, account_id),
        "create-rds": lambda: create_rds(clients, account_id),
        "populate":   lambda: populate_rds(clients, account_id),
        "upload":     lambda: upload_ml_files(clients, account_id),
        "delete":     lambda: cmd_delete(clients, account_id),
        "delete-s3":  lambda: delete_s3(clients, account_id),
        "delete-rds": lambda: delete_rds(clients),
    }

    dispatch[command]()
    logger.info("\n✅ Done")


if __name__ == "__main__":
    main()