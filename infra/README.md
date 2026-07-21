# FireWorks WebGUI — Secure AWS Deployment

A read-only monitoring dashboard for the Impulse/FireWorks LaunchPad, hosted on a
single hardened EC2 instance in `us-east-1`, fronted by Caddy for automatic HTTPS.

## Architecture

```
Internet ──HTTPS(443)──> Caddy (Let's Encrypt via sslip.io)
                           │ reverse_proxy → 127.0.0.1:8080
                           ▼
                  gunicorn + FireWorks Flask WebGUI (Docker)
                           │ pymongo+srv (READ-ONLY Atlas user)
                           ▼
                 MongoDB Atlas (impulse-prod-1, db: fireworks)
```

- **Compute**: 1× EC2 `t3.small`, Amazon Linux 2023, encrypted gp3 EBS, IMDSv2 required.
- **TLS**: Caddy auto-provisions a valid Let's Encrypt cert for `fireworks.<eip-dashed>.sslip.io`.
- **AuthN**: FireWorks HTTP Basic Auth on every route.
- **Secrets**: AWS Secrets Manager, fetched at boot via a least-privilege IAM instance role.
- **DB access**: a **read-only** MongoDB Atlas user — the GUI cannot modify any data.

## Security properties

| Control | Implementation |
|---|---|
| Read-only data access | Atlas DB user with built-in `read` role on `fireworks` db |
| No plaintext creds in git/state | Secrets Manager; values set out-of-band via CLI |
| Least-privilege IAM | Instance role can only `secretsmanager:GetSecretValue` on the 2 secrets |
| Encryption at rest | gp3 EBS `encrypted = true` |
| Instance metadata hardening | IMDSv2 (`http_tokens = required`) |
| Network exposure | SG allows 443/80 public, 22 from operator IP only |
| Transport security | Caddy HTTPS (valid public cert) |

## One-time manual step: create the read-only Atlas user

1. Atlas → **Database Access** → Add New Database User.
   - Authentication: Password.
   - Built-in Role: **Only read any database** (or `read` scoped to the `fireworks` db).
   - Username e.g. `monitoring_ro`, generate a strong password.
2. Atlas → **Network Access** → add the instance Elastic IP (see `terraform output elastic_ip`), `/32`.
3. Store the read-only SRV URI in Secrets Manager (see below).

## Deploy

```bash
cd infra
terraform init
terraform apply
```

## Bootstrapping remote state (one-time)

State currently lives locally in `infra/terraform.tfstate`, which is git-ignored
and therefore not shared between operators. To move it to a shared, versioned,
encrypted S3 backend with DynamoDB-based locking:

```bash
# 1. Create the state bucket (versioning + SSE-S3 + block public access).
aws s3api create-bucket \
  --region us-east-1 \
  --bucket impulse-terraform-state-548317354126

aws s3api put-bucket-versioning \
  --bucket impulse-terraform-state-548317354126 \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket impulse-terraform-state-548317354126 \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block \
  --bucket impulse-terraform-state-548317354126 \
  --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

# 2. Create the state-lock table.
aws dynamodb create-table \
  --region us-east-1 \
  --table-name impulse-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 3. Uncomment the `backend "s3"` block in main.tf.
# 4. Migrate local state into the bucket.
cd infra
terraform init -migrate-state
# Terraform will prompt: "Do you want to copy existing state to the new backend?" -> yes

# 5. After migration succeeds, the local terraform.tfstate is no longer authoritative.
#    It's already git-ignored; safe to delete or leave in place.
rm -f terraform.tfstate terraform.tfstate.backup
```

After bootstrapping, every operator running `terraform init` in this directory
will read/write the shared state in S3, with DynamoDB preventing concurrent
applies from stepping on each other.

### Set secret values (out-of-band, never in TF state)

```bash
# Read-only Atlas connection string
aws secretsmanager put-secret-value --region us-east-1 \
  --secret-id impulse/fireworks-webgui/mongodb-uri \
  --secret-string 'mongodb+srv://monitoring_ro:PASSWORD@impulse-prod-1.pfjqa.mongodb.net/'

# Basic-auth credentials (username/password for the dashboard)
aws secretsmanager put-secret-value --region us-east-1 \
  --secret-id impulse/fireworks-webgui/basic-auth \
  --secret-string '{"username":"monitoringuser","password":"CHANGEME"}'
```

If you set/rotate secrets after the instance is already running, re-run the
bootstrap by tainting the instance so `user_data` re-executes:

```bash
terraform apply -replace=aws_instance.webgui
```

## Access

- URL: `terraform output webgui_url` → e.g. `https://fireworks.54-89-29-96.sslip.io`
- Log in with the Basic Auth username/password.

## Operations

- **Admin shell** (no SSH key needed): `aws ssm start-session --target <instance_id>`
- **Logs**: `sudo cat /var/log/cloud-init-output.log` (bootstrap), `docker logs fireworks-webgui`, `journalctl -u caddy`.
- **Restart app**: `docker restart fireworks-webgui`
- **Rotate basic auth**: update the `basic-auth` secret, then `terraform apply -replace=aws_instance.webgui`.

## Teardown

```bash
cd infra && terraform destroy
```
Secrets are retained for a recovery window by default; force-delete with
`aws secretsmanager delete-secret --force-delete-without-recovery` if needed.
