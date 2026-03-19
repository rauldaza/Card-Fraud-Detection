"""
ft_MLOps_pipeline.py — Config-Driven SageMaker Pipeline
========================================================
Builds the full fraud-detection pipeline by looping over the ``models``
array in config.json.  Each model entry defines its own scripts, S3
prefixes, and training hyperparameters so adding a new model is a pure
config change — no pipeline code modifications required.

Pipeline DAG:
  SplitData
     ├── TabularTransformer_Preprocessing → TabularTransformer_Training → TabularTransformer_Evaluation
     └── FTTransformer_Preprocessing      → FTTransformer_Training      → FTTransformer_Evaluation

Usage:
  python ft_MLOps_pipeline.py

Which models run is determined entirely by the ``models`` array in config.json.
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

from MLOps_pipeline import (
    load_config,
    build_paths,
    setup_vpc_networking,
    build_split_args,
)

# ─── Test / Full-run Toggle ──────────────────────────────────────────────────
TEST_MODE = True   # Set to False for a full training run
# ─────────────────────────────────────────────────────────────────────────────


def build_model_s3_paths(cfg: dict, model_cfg: dict) -> dict:
    """
    Build fully-qualified S3 URIs for a single model's artifacts.

    Parameters
    ----------
    cfg : dict
        Top-level pipeline config (needs ``s3.models_bucket_prefix``; AWS
        account ID is derived dynamically from boto3 STS).
    model_cfg : dict
        One element from ``cfg["models"]``, containing ``s3_prefixes``.

    Returns
    -------
    dict
        Keys: ``s3_pkl``, ``s3_output``, ``s3_eval`` — full S3 URIs.
    """
    account       = boto3.client("sts").get_caller_identity()["Account"]
    models_bucket = f"{cfg['s3']['models_bucket_prefix']}-{account}"
    base          = f"s3://{models_bucket}"
    prefixes      = model_cfg["s3_prefixes"]

    return {
        "s3_pkl":    f"{base}/{prefixes['pkl']}",
        "s3_output": f"{base}/{prefixes['output']}",
        "s3_eval":   f"{base}/{prefixes['eval']}",
    }


def build_training_args_from_config(model_cfg: dict, test_mode: bool) -> list[str]:
    """
    Build CLI arguments for a model's training script from its config entry.

    Always emits ``--epochs``, ``--batch-size``, and ``--target-col``.
    Any **additional** keys in the ``full_run`` / ``test_run`` dict are
    flattened to ``--key-name value`` (underscores → hyphens), so
    model-specific args like ``--embed-dim`` or ``--d-model`` require
    zero code changes — just add them to the config.

    Parameters
    ----------
    model_cfg : dict
        One element from ``cfg["models"]``.
    test_mode : bool
        If True, use ``training_args.test_run``; otherwise ``full_run``.

    Returns
    -------
    list[str]
        Flat list of CLI argument strings.
    """
    t = model_cfg["training_args"]
    run_cfg = t["test_run"] if test_mode else t["full_run"]

    args = [
        "--epochs",     str(run_cfg["epochs"]),
        "--batch-size", str(t["batch_size"]),
        "--target-col", t["target_col"],
    ]

    # Flatten remaining keys (skip "epochs" — already handled)
    for key, value in run_cfg.items():
        if key == "epochs":
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        args += [cli_flag, str(value)]

    return args


def build_model_branch(
    model_cfg: dict,
    model_s3: dict,
    cfg: dict,
    pretrain_processor: ScriptProcessor,
    train_processor: ScriptProcessor,
    split_train_uri,
    split_test_uri,
    test_mode: bool,
    depends_on_steps: list = None,
) -> list[ProcessingStep]:
    """
    Build the three pipeline steps (preprocess → train → eval) for one model.

    Parameters
    ----------
    model_cfg : dict
        One element from ``cfg["models"]``.
    model_s3 : dict
        S3 URIs from ``build_model_s3_paths()``.
    cfg : dict
        Top-level pipeline config.
    pretrain_processor : ScriptProcessor
        CPU processor (used for preprocessing).
    train_processor : ScriptProcessor
        GPU processor (used for training and evaluation).
    split_train_uri : sagemaker.workflow.properties.Properties
        Lazy S3 reference to the training split output.
    split_test_uri : sagemaker.workflow.properties.Properties
        Lazy S3 reference to the test split output.
    test_mode : bool
        If True, use test-run hyperparameters.
    depends_on_steps : list, optional
        A list of step names that this model's preprocessing step must wait for.

    Returns
    -------
    list[ProcessingStep]
        Three steps: [preprocessing, training, evaluation].
    """
    name    = model_cfg["name"]
    scripts = model_cfg["scripts"]

    # ── Preprocessing ────────────────────────────────────────────────────────
    step_preprocess = ProcessingStep(
        name=f"{name}_Preprocessing",
        depends_on=depends_on_steps,
        step_args=pretrain_processor.run(
            code=scripts["preprocessing"],
            arguments=["--target", cfg["data_split"]["target_col"]],
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
                    destination=model_s3["s3_pkl"],
                ),
            ],
        ),
    )

    pkl_uri = (
        step_preprocess.properties
        .ProcessingOutputConfig.Outputs["preprocessor"].S3Output.S3Uri
    )

    # ── Training ─────────────────────────────────────────────────────────────
    training_args = build_training_args_from_config(model_cfg, test_mode)

    step_train = ProcessingStep(
        name=f"{name}_Training",
        step_args=train_processor.run(
            code=scripts["train"],
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
                    destination=model_s3["s3_output"],
                ),
            ],
        ),
    )

    model_uri = (
        step_train.properties
        .ProcessingOutputConfig.Outputs["model-checkpoint"].S3Output.S3Uri
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    step_eval = ProcessingStep(
        name=f"{name}_Evaluation",
        step_args=train_processor.run(
            code=scripts["test"],
            arguments=[
                "--test-data",  "/opt/ml/processing/input/test/transactions_test.feather",
                "--model-dir",  "/opt/ml/processing/input/model",
                "--config-dir", "/opt/ml/processing/input/config",
                "--target-col", model_cfg["training_args"]["target_col"],
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
                    source=model_uri,
                    destination="/opt/ml/processing/input/model",
                    input_name="model-checkpoint",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-output",
                    source="/opt/ml/processing/output",
                    destination=model_s3["s3_eval"],
                ),
            ],
        ),
    )

    return [step_preprocess, step_train, step_eval]


def build_pipeline(
    cfg: dict,
    paths: dict,
    vpc: dict,
    split_args: list[str],
    pipeline_name: str,
    test_mode: bool,
) -> tuple[Pipeline, boto3.Session]:
    """
    Construct the SageMaker pipeline: shared split + N model branches.

    Parameters
    ----------
    cfg : dict
        Top-level pipeline config.
    paths : dict
        Shared S3 paths from ``build_paths()``.
    vpc : dict
        VPC networking info from ``setup_vpc_networking()``.
    split_args : list[str]
        CLI arguments for ``data_split.py``.
    pipeline_name : str
        Name for the SageMaker pipeline.
    test_mode : bool
        If True, use test-run hyperparameters.

    Returns
    -------
    tuple[Pipeline, boto3.Session]
    """
    # ── AWS Sessions ──────────────────────────────────────────────────────────
    boto_session     = boto3.Session(
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
    # Step 1: SplitData — shared by all model branches
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
    # Model branches — one per entry in selected_models
    # ══════════════════════════════════════════════════════════════════════════
    all_steps = [step_split]
    
    branch_terminating_steps_names = []
    max_concurrency = cfg.get("max_concurrent_models", 1)

    for i, model_cfg in enumerate(cfg["models"]):
        model_s3 = build_model_s3_paths(cfg, model_cfg)
        
        depends_on = None
        if i >= max_concurrency:
            depends_on = [branch_terminating_steps_names[i - max_concurrency]]

        branch_steps = build_model_branch(
            model_cfg, model_s3, cfg,
            pretrain_processor, train_processor,
            split_train_uri, split_test_uri,
            test_mode,
            depends_on_steps=depends_on,
        )
        
        branch_terminating_steps_names.append(branch_steps[-1].name)
        all_steps.extend(branch_steps)

    return Pipeline(
        name=pipeline_name,
        steps=all_steps,
        sagemaker_session=pipeline_session,
    ), boto_session


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    cfg   = load_config(os.path.join(os.path.dirname(__file__), "config.json"))
    paths = build_paths(cfg)

    # ── AWS Clients ───────────────────────────────────────────────────────────
    boto_session = boto3.Session(
        region_name=cfg["aws"]["region"],
    )
    ec2_client = boto_session.client("ec2")
    rds_client = boto_session.client("rds")

    # ── VPC Networking ────────────────────────────────────────────────────────
    vpc = setup_vpc_networking(ec2_client, rds_client, cfg, cfg["aws"]["region"])

    # ── Build arguments ───────────────────────────────────────────────────────
    split_args = build_split_args(cfg, vpc, TEST_MODE)

    suffix        = "TEST" if TEST_MODE else "FULL"
    model_names   = [m["name"] for m in cfg["models"]]
    pipeline_name = f"FraudDetection-{suffix}"

    # ── Print run summary ─────────────────────────────────────────────────────
    mode_label = "TEST MODE" if TEST_MODE else "FULL RUN"

    print(f"\n{'='*60}")
    print(f"  {mode_label}  |  Models: {', '.join(model_names)}")
    print(f"{'='*60}")
    print(f"  Pretrain instance: {cfg['pretrain_compute']['instance_type']} (CPU + VPC)")
    print(f"  Train instance:    {cfg['train_compute']['instance_type']} (GPU)")
    print(f"  Pipeline:          {pipeline_name}")
    print(f"  RDS Host:          {vpc['rds_host']}")
    print(f"{'='*60}\n")

    # ── Validate local scripts ────────────────────────────────────────────────
    scripts = {"steps/data_split.py"}
    for m in cfg["models"]:
        scripts.update(m["scripts"].values())

    for script in sorted(scripts):
        if not os.path.exists(script):
            raise FileNotFoundError(f"Required script not found: {script}")

    # ── Build, register, and start pipeline ───────────────────────────────────
    pipeline, boto_session = build_pipeline(
        cfg, paths, vpc, split_args,
        pipeline_name, TEST_MODE,
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
