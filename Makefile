# ─────────────────────────────────────────────────────────────
#  Makefile — Build & push mlops-pipeline images to ECR
#
#  Two images, two ECR repos:
#    mlops-pipeline-pretrain  (CPU: SplitData + TabPreprocessing)
#    mlops-pipeline-train     (GPU: Train + Eval)
#
#  Commands:
#    make all          →  build + login + push both images  (most common)
#    make build        →  build both images locally
#    make build-pre    →  build pretrain image only
#    make build-train  →  build train image only
#    make login        →  authenticate Docker to ECR
#    make push         →  push both images to ECR
#    make push-pre     →  push pretrain image only
#    make push-train   →  push train image only
#    make clean        →  remove both local images
# ─────────────────────────────────────────────────────────────

REGION    := us-east-1
IMAGE_TAG := latest

# Account ID resolved from the default credential chain (no profile)
ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)

ECR_BASE := $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com

# ── Pretrain image (CPU — SplitData + TabPreprocessing) ──────
PRE_LOCAL      := mlops-pipeline-pretrain
PRE_REPO       := mlops-pipeline-pretrain
PRE_DOCKERFILE := DockerImages/Pretrain.dockerfile
PRE_ECR_URI    := $(ECR_BASE)/$(PRE_REPO)

# ── Train/Eval image (GPU — Train + Eval) ────────────────────
TRAIN_LOCAL      := mlops-pipeline-train
TRAIN_REPO       := mlops-pipeline-train
TRAIN_DOCKERFILE := DockerImages/TrainEval.dockerfile
TRAIN_ECR_URI    := $(ECR_BASE)/$(TRAIN_REPO)

.PHONY: all build build-pre build-train login push push-pre push-train clean

# ── Composite targets ─────────────────────────────────────────

## Build + login + push both images
all: build login push

## Build both images locally
build: build-pre build-train

## Push both images to ECR
push: push-pre push-train

# ── Pretrain image ────────────────────────────────────────────

build-pre:
	@echo ""
	@echo "Building pretrain image: $(PRE_LOCAL):$(IMAGE_TAG)"
	docker build \
	  -f $(PRE_DOCKERFILE) \
	  -t $(PRE_LOCAL):$(IMAGE_TAG) \
	  .
	@echo "Pretrain build complete"

push-pre:
	@echo ""
	@echo "Pushing pretrain image to $(PRE_ECR_URI):$(IMAGE_TAG)"
	docker tag $(PRE_LOCAL):$(IMAGE_TAG) $(PRE_ECR_URI):$(IMAGE_TAG)
	docker push $(PRE_ECR_URI):$(IMAGE_TAG)
	@echo "Pretrain pushed: $(PRE_ECR_URI):$(IMAGE_TAG)"

# ── Train/Eval image ──────────────────────────────────────────

build-train:
	@echo ""
	@echo "Building train/eval image: $(TRAIN_LOCAL):$(IMAGE_TAG)"
	docker build \
	  -f $(TRAIN_DOCKERFILE) \
	  -t $(TRAIN_LOCAL):$(IMAGE_TAG) \
	  .
	@echo "Train/eval build complete"

push-train:
	@echo ""
	@echo "Pushing train/eval image to $(TRAIN_ECR_URI):$(IMAGE_TAG)"
	docker tag $(TRAIN_LOCAL):$(IMAGE_TAG) $(TRAIN_ECR_URI):$(IMAGE_TAG)
	docker push $(TRAIN_ECR_URI):$(IMAGE_TAG)
	@echo "Train/eval pushed: $(TRAIN_ECR_URI):$(IMAGE_TAG)"

# ── Shared ────────────────────────────────────────────────────

## Authenticate local Docker daemon to ECR
login:
	@echo ""
	@echo "Logging in to ECR (region: $(REGION))..."
	aws ecr get-login-password \
	  --region $(REGION) \
	| docker login \
	  --username AWS \
	  --password-stdin \
	  $(ECR_BASE)
	@echo "Logged in to $(ECR_BASE)"

## Remove both local images (ECR copies are unaffected)
clean:
	@echo "Removing local images..."
	docker rmi $(PRE_LOCAL):$(IMAGE_TAG)     || true
	docker rmi $(TRAIN_LOCAL):$(IMAGE_TAG)   || true
	docker rmi $(PRE_ECR_URI):$(IMAGE_TAG)   || true
	docker rmi $(TRAIN_ECR_URI):$(IMAGE_TAG) || true
	@echo "Clean complete"
