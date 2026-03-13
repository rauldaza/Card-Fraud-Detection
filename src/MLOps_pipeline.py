"""
MLOps_pipeline.py — SageMaker Fraud Detection End-to-End Pipeline
=================================================================
Four-step pipeline connecting pretrain and train-eval:
  1. SplitData             – reads from RDS, writes train/test feather to S3
  2. TabPreprocessing      – fits sklearn preprocessor, saves .pkl to S3
  3. TrainTabularTransformer – trains the model on GPU
  4. EvaluateTabularTransformer – evaluates the model, logs to MLflow

Reads all configuration from config.json.
Toggle TEST_MODE below to switch between a quick validation run and a full run.
"""

import json
import os

import boto3
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

# ─── Test / Full-run Toggle ──────────────────────────────────────────────────
TEST_MODE = True   # Set to False for a full 10-epoch training run
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str = "config.json") -> dict:
    """Load pipeline configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def build_paths(cfg: dict) -> dict:
    """Derive all S3 URIs and ECR image URIs from the config.

    Every derived value uses aws.account_id so there is only one
    place to change when switching AWS accounts.
    """
    account = cfg["aws"]["account_id"]
    region  = cfg["aws"]["region"]

    data_bucket   = f"{cfg['s3']['data_bucket_prefix']}-{account}"
    models_bucket = f"{cfg['s3']['models_bucket_prefix']}-{account}"

    data   = f"s3://{data_bucket}"
    models = f"s3://{models_bucket}"

    return {
        # Resolved role ARN
        "role_arn": f"arn:aws:iam::{account}:role/{cfg['aws']['role_name']}",
        # Docker images
        "pretrain_image": (
            f"{account}.dkr.ecr.{region}.amazonaws.com"
            f"/{cfg['ecr']['pretrain_repo']}:{cfg['ecr']['pretrain_tag']}"
        ),
        "train_image": (
            f"{account}.dkr.ecr.{region}.amazonaws.com"
            f"/{cfg['ecr']['train_repo']}:{cfg['ecr']['train_tag']}"
        ),
        # S3 paths — pretrain
        "s3_split_train": f"{data}/{cfg['s3']['split_train_key']}",
        "s3_split_test":  f"{data}/{cfg['s3']['split_test_key']}",
        "s3_pkl":         f"{models}/{cfg['s3']['pkl_key']}",
        # S3 paths — train/eval
        # s3_train / s3_test removed: Steps 3+4 use lazy split URIs from Step 1
        "s3_output":      f"{models}/{cfg['s3']['output_prefix']}",
        "s3_eval":        f"{models}/{cfg['s3']['eval_prefix']}",
    }


def setup_vpc_networking(ec2_client, rds_client, cfg: dict, region: str) -> dict:
    """
    Fetch RDS networking info and ensure the VPC is ready for SageMaker:
      - Self-referencing inbound rule on the RDS port
      - S3 Gateway Endpoint (SageMaker in a VPC has no internet access)

    Returns a dict with keys: subnets, security_group_ids, rds_host, rds_port.
    """
    # ── Fetch RDS instance details ────────────────────────────────────────────
    db_instance = rds_client.describe_db_instances(
        DBInstanceIdentifier=cfg["rds"]["db_identifier"]
    )["DBInstances"][0]

    rds_vpc_id = db_instance["DBSubnetGroup"]["VpcId"]
    rds_subnets = [
        s["SubnetIdentifier"]
        for s in db_instance["DBSubnetGroup"]["Subnets"]
    ]
    rds_security_group_ids = [
        sg["VpcSecurityGroupId"]
        for sg in db_instance["VpcSecurityGroups"]
        if sg["Status"] == "active"
    ]

    # ── Self-referencing SG rule ──────────────────────────────────────────────
    rds_port = cfg["rds"]["port"]
    for sg_id in rds_security_group_ids:
        try:
            ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "tcp",
                    "FromPort":   rds_port,
                    "ToPort":     rds_port,
                    "UserIdGroupPairs": [{"GroupId": sg_id}],
                }],
            )
            print(f"[VPC] Added self-referencing inbound rule on port {rds_port} to {sg_id}")
        except ec2_client.exceptions.ClientError as e:
            if "Duplicate" in str(e) or "already exists" in str(e):
                print(f"[VPC] Self-referencing rule already exists on {sg_id}")
            else:
                raise

    # ── S3 Gateway Endpoint ───────────────────────────────────────────────────
    s3_service_name = f"com.amazonaws.{region}.s3"
    existing_endpoints = ec2_client.describe_vpc_endpoints(
        Filters=[
            {"Name": "vpc-id",       "Values": [rds_vpc_id]},
            {"Name": "service-name", "Values": [s3_service_name]},
        ]
    )["VpcEndpoints"]

    if not existing_endpoints:
        route_tables = ec2_client.describe_route_tables(
            Filters=[{"Name": "vpc-id", "Values": [rds_vpc_id]}]
        )["RouteTables"]
        route_table_ids = [rt["RouteTableId"] for rt in route_tables]

        ec2_client.create_vpc_endpoint(
            VpcId=rds_vpc_id,
            ServiceName=s3_service_name,
            VpcEndpointType="Gateway",
            RouteTableIds=route_table_ids,
        )
        print(f"[VPC] Created S3 Gateway Endpoint for {rds_vpc_id}")
    else:
        print("[VPC] S3 Gateway Endpoint already exists")

    print(f"[VPC] RDS VPC: {rds_vpc_id}")
    print(f"[VPC] Subnets: {rds_subnets}")
    print(f"[VPC] Security Groups: {rds_security_group_ids}")

    return {
        "subnets":            rds_subnets,
        "security_group_ids": rds_security_group_ids,
        "rds_host":           db_instance["Endpoint"]["Address"],
        "rds_port":           str(db_instance["Endpoint"]["Port"]),
    }


def build_split_args(cfg: dict, vpc: dict, test_mode: bool) -> list[str]:
    """Return the command-line arguments passed to data_split.py."""
    ds = cfg["data_split"]
    args = [
        "--db-host",      vpc["rds_host"],
        "--db-port",      vpc["rds_port"],
        "--db-user",      cfg["rds"]["db_user"],
        "--db-password",  cfg["rds"]["db_password"],
        "--target",       ds["target_col"],
        "--id-col",       ds["id_col"],
        "--test-size",    str(ds["test_size"]),
        "--random-state", str(ds["random_state"]),
    ]
    if not ds["stratify"]:
        args.append("--no-stratify")
    if test_mode:
        args += ["--limit", str(cfg["test_run"]["limit"])]
    return args


def build_training_args(cfg: dict, test_mode: bool) -> list[str]:
    """Return the command-line arguments passed to train.py."""
    run_cfg = cfg["training"]["test_run"] if test_mode else cfg["training"]["full_run"]
    args = [
        "--epochs",     str(run_cfg["epochs"]),
        "--batch-size", str(cfg["training"]["batch_size"]),
        "--target-col", cfg["training"]["target_col"],
    ]
    if not test_mode:
        args += [
            "--learning-rate", str(run_cfg["learning_rate"]),
            "--embed-dim",     str(run_cfg["embed_dim"]),
        ]
    return args


def build_pipeline(
    cfg: dict, paths: dict, vpc: dict,
    split_args: list[str], training_args: list[str],
    pipeline_name: str,
) -> tuple[Pipeline, boto3.Session]:
    """Construct the four-step SageMaker pipeline."""

    # ── AWS Sessions ──────────────────────────────────────────────────────────
    boto_session     = boto3.Session(
        profile_name=cfg["aws"]["profile"],
        region_name=cfg["aws"]["region"],
    )
    pipeline_session = PipelineSession(boto_session=boto_session)

    pretrain = cfg["pretrain_compute"]
    train    = cfg["train_compute"]

    # ── Processor 1: Pretrain (CPU + VPC for RDS) ─────────────────────────────
    pretrain_processor = ScriptProcessor(
        image_uri              = paths["pretrain_image"],
        role                   = paths["role_arn"],
        instance_count         = pretrain["instance_count"],
        instance_type          = pretrain["instance_type"],
        command                = ["python3"],
        sagemaker_session      = pipeline_session,
        volume_size_in_gb      = pretrain["volume_size_gb"],
        max_runtime_in_seconds = pretrain["max_runtime_seconds"],
        base_job_name          = "fraud-pretrain",
        network_config         = NetworkConfig(
            enable_network_isolation=False,
            subnets=vpc["subnets"],
            security_group_ids=vpc["security_group_ids"],
        ),
    )

    # ── Processor 2: Train/Eval (GPU, no VPC) ─────────────────────────────────
    # FIX: was cfg["aws"]["role"] (wrong key, inconsistent with everywhere else).
    # FIX: added max_runtime_in_seconds — essential for long GPU training jobs.
    train_processor = ScriptProcessor(
        image_uri              = paths["train_image"],
        role                   = paths["role_arn"],
        instance_count         = train["instance_count"],
        instance_type          = train["instance_type"],
        command                = ["python3"],
        sagemaker_session      = pipeline_session,
        volume_size_in_gb      = train["volume_size_gb"],
        max_runtime_in_seconds = train["max_runtime_seconds"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: SplitData — read from RDS, write train/test feather to S3
    # ══════════════════════════════════════════════════════════════════════════
    step_split = ProcessingStep(
        name="SplitData",
        step_args=pretrain_processor.run(
            code="steps/data_split.py",
            arguments=split_args,
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train",
                    destination=paths["s3_split_train"],
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test",
                    destination=paths["s3_split_test"],
                ),
            ],
        ),
    )

    # Lazy S3 references — resolved at pipeline runtime
    split_train_uri = (
        step_split.properties
        .ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    )
    split_test_uri = (
        step_split.properties
        .ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: TabPreprocessing — fit preprocessor on train split
    # ══════════════════════════════════════════════════════════════════════════
    step_preprocess = ProcessingStep(
        name="TabPreprocessing",
        step_args=pretrain_processor.run(
            code="steps/tab_preprocessing.py",
            arguments=[
                "--target", cfg["data_split"]["target_col"],
            ],
            inputs=[
                ProcessingInput(
                    source=split_train_uri,
                    destination="/opt/ml/processing/train",
                    input_name="train-data",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="preprocessor",
                    source="/opt/ml/processing/artifacts",
                    destination=paths["s3_pkl"],
                ),
            ],
        ),
    )

    # Lazy reference to the .pkl output
    pkl_uri = (
        step_preprocess.properties
        .ProcessingOutputConfig.Outputs["preprocessor"].S3Output.S3Uri
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: TrainTabularTransformer — train the model on GPU
    # ══════════════════════════════════════════════════════════════════════════
    step_train = ProcessingStep(
        name="TrainTabularTransformer",
        step_args=train_processor.run(
            code="steps/train.py",
            arguments=training_args,
            inputs=[
                ProcessingInput(
                    source=split_train_uri,
                    destination="/opt/ml/processing/input/train",
                    input_name="train-data",
                ),
                ProcessingInput(
                    source=split_test_uri,
                    destination="/opt/ml/processing/input/test",
                    input_name="test-data",
                ),
                ProcessingInput(
                    source=pkl_uri,
                    destination="/opt/ml/processing/input/config",
                    input_name="preprocessing-config",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="model-checkpoint",
                    source="/opt/ml/processing/output",
                    destination=paths["s3_output"],
                ),
            ],
        ),
    )

    # Lazy reference to the model checkpoint
    model_s3_uri = (
        step_train.properties
        .ProcessingOutputConfig.Outputs["model-checkpoint"].S3Output.S3Uri
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: EvaluateTabularTransformer — evaluate model, log to MLflow
    # ══════════════════════════════════════════════════════════════════════════
    step_eval = ProcessingStep(
        name="EvaluateTabularTransformer",
        step_args=train_processor.run(
            code="steps/test.py",
            arguments=[
                "--test-data",  "/opt/ml/processing/input/test/transactions_test.feather",
                "--model-dir",  "/opt/ml/processing/input/model",
                "--config-dir", "/opt/ml/processing/input/config",
                "--target-col", cfg["training"]["target_col"],
            ],
            inputs=[
                ProcessingInput(
                    source=split_test_uri,
                    destination="/opt/ml/processing/input/test",
                    input_name="test-data",
                ),
                ProcessingInput(
                    source=pkl_uri,
                    destination="/opt/ml/processing/input/config",
                    input_name="preprocessing-config",
                ),
                ProcessingInput(
                    source=model_s3_uri,
                    destination="/opt/ml/processing/input/model",
                    input_name="model-checkpoint",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-output",
                    source="/opt/ml/processing/output",
                    destination=paths["s3_eval"],
                ),
            ],
        ),
    )

    return Pipeline(
        name=pipeline_name,
        steps=[step_split, step_preprocess, step_train, step_eval],
        sagemaker_session=pipeline_session,
    ), boto_session


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    cfg   = load_config(os.path.join(os.path.dirname(__file__), "config.json"))
    paths = build_paths(cfg)

    # ── AWS Clients ───────────────────────────────────────────────────────────
    boto_session = boto3.Session(
        profile_name=cfg["aws"]["profile"],
        region_name=cfg["aws"]["region"],
    )
    ec2_client = boto_session.client("ec2")
    rds_client = boto_session.client("rds")

    # ── VPC Networking ────────────────────────────────────────────────────────
    vpc = setup_vpc_networking(ec2_client, rds_client, cfg, cfg["aws"]["region"])

    # ── Build arguments ───────────────────────────────────────────────────────
    split_args    = build_split_args(cfg, vpc, TEST_MODE)
    training_args = build_training_args(cfg, TEST_MODE)
    pipeline_name = "FraudDetection-TEST" if TEST_MODE else "FraudDetectionPipeline"

    # ── Print run summary ─────────────────────────────────────────────────────
    mode_label = "TEST MODE" if TEST_MODE else "FULL RUN"
    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    print(f"{'='*60}")
    print(f"  Pretrain instance: {cfg['pretrain_compute']['instance_type']} (CPU + VPC)")
    print(f"  Train instance:    {cfg['train_compute']['instance_type']} (GPU)")
    print(f"  Pipeline:          {pipeline_name}")
    print(f"  RDS Host:          {vpc['rds_host']}")
    if TEST_MODE:
        print(f"  Row Limit:         {cfg['test_run']['limit']}")
        print(f"  Epochs:            {cfg['training']['test_run']['epochs']}")
    else:
        print(f"  Epochs:            {cfg['training']['full_run']['epochs']}")
    print(f"{'='*60}\n")

    # ── Validate local scripts exist ──────────────────────────────────────────
    scripts = [
        "steps/data_split.py",
        "steps/tab_preprocessing.py",
        "steps/train.py",
        "steps/test.py",
    ]
    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Required script not found: {script}")

    # ── Build, register, and start pipeline ───────────────────────────────────
    pipeline, boto_session = build_pipeline(
        cfg, paths, vpc, split_args, training_args, pipeline_name
    )

    print(f"Registering pipeline '{pipeline_name}' ...")
    pipeline.upsert(role_arn=paths["role_arn"])

    execution = pipeline.start()
    print(f"Pipeline started  : {execution.arn}")
    print(f"\nMonitor at:")
    print(f"  AWS Console → SageMaker → Pipelines → {pipeline_name}")

    # ── Test-mode cleanup ─────────────────────────────────────────────────────
    if TEST_MODE:
        input("\n>>> Press ENTER to delete the pipeline definition (S3 outputs kept)...")
        boto_session.client("sagemaker").delete_pipeline(PipelineName=pipeline_name)
        print("Pipeline definition deleted. S3 outputs are NOT deleted.")


if __name__ == "__main__":
    main()