terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.52.0"
    }
  }

  # Remote state: S3 for storage, DynamoDB for locking.
  # Bootstrap steps (one-time, per README.md "Bootstrapping remote state"):
  #   1. Create the bucket + DynamoDB table manually with the AWS CLI.
  #   2. Uncomment this block.
  #   3. Run `terraform init -migrate-state` to move any local state up.
  # Until step 1 is complete, leave this block commented out.
  #
  # backend "s3" {
  #   bucket         = "impulse-terraform-state-548317354126"
  #   key            = "fireworks-webgui/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "impulse-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region
}

# ─────────────────────────────────────────────────────────────────────────────
# Variables
# ─────────────────────────────────────────────────────────────────────────────

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC to deploy into (defaults to the account default VPC)"
  type        = string
  default     = "vpc-7f745e19"
}

variable "subnet_id" {
  description = "Public subnet (auto-assign public IP) to launch the instance in"
  type        = string
  default     = "subnet-9b26e2d3"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to reach SSH (22). Defaults to the operator's current IP."
  type        = string
  default     = "129.105.121.183/32"
}

variable "project" {
  description = "Name/tag prefix"
  type        = string
  default     = "impulse-fireworks-webgui"
}

# ─────────────────────────────────────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────────────────────────────────────

data "aws_vpc" "selected" {
  id = var.vpc_id
}

# Always pull the latest Amazon Linux 2023 AMI rather than pinning a stale id.
data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# ─────────────────────────────────────────────────────────────────────────────
# Secrets Manager — values are set out-of-band (never in TF state/git)
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "mongodb_uri" {
  name        = "impulse/fireworks-webgui/mongodb-uri"
  description = "Read-only MongoDB Atlas SRV connection string for the FireWorks WebGUI"
}

resource "aws_secretsmanager_secret" "basic_auth" {
  name        = "impulse/fireworks-webgui/basic-auth"
  description = "HTTP Basic Auth credentials (JSON: username/password) for the FireWorks WebGUI"
}

# ─────────────────────────────────────────────────────────────────────────────
# IAM — least-privilege instance role: read only the two secrets above
# ─────────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "webgui" {
  name               = "${var.project}-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "secrets_read" {
  statement {
    sid     = "ReadWebguiSecrets"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.mongodb_uri.arn,
      aws_secretsmanager_secret.basic_auth.arn,
    ]
  }
}

resource "aws_iam_role_policy" "secrets_read" {
  name   = "${var.project}-secrets-read"
  role   = aws_iam_role.webgui.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# Allow Session Manager access so no SSH key is strictly required for admin.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.webgui.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "webgui" {
  name = "${var.project}-profile"
  role = aws_iam_role.webgui.name
}

# ─────────────────────────────────────────────────────────────────────────────
# Security group
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_security_group" "webgui" {
  name        = "${var.project}-sg"
  description = "FireWorks WebGUI: HTTPS/HTTP public, SSH restricted"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP (ACME challenge + redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH (operator only)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    description = "All outbound (Atlas TLS, ACME, package install)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-sg" }
}

# ─────────────────────────────────────────────────────────────────────────────
# EC2 instance
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_instance" "webgui" {
  ami                    = data.aws_ssm_parameter.al2023_ami.value
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.webgui.id]
  iam_instance_profile   = aws_iam_instance_profile.webgui.name

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region          = var.region
    mongodb_secret  = aws_secretsmanager_secret.mongodb_uri.name
    basicauth_secret = aws_secretsmanager_secret.basic_auth.name
    eip             = aws_eip.webgui.public_ip
  })

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  tags = { Name = var.project }
}

resource "aws_eip" "webgui" {
  domain = "vpc"
  tags   = { Name = "${var.project}-eip" }
}

resource "aws_eip_association" "webgui" {
  instance_id   = aws_instance.webgui.id
  allocation_id = aws_eip.webgui.id
}

# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────

output "elastic_ip" {
  description = "Public Elastic IP — allowlist this in MongoDB Atlas Network Access"
  value       = aws_eip.webgui.public_ip
}

output "webgui_url" {
  description = "HTTPS URL for the FireWorks WebGUI"
  value       = "https://fireworks.${replace(aws_eip.webgui.public_ip, ".", "-")}.sslip.io"
}

output "instance_id" {
  value = aws_instance.webgui.id
}
