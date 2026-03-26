"""Storage stack: S3 bucket and ECR repositories."""

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_ecr as ecr


class StorageStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ── S3 bucket for all document I/O ───────────────────────────────

        self.bucket = s3.Bucket(
            self,
            "ImpulseBucket",
            bucket_name=cdk.Fn.select(
                0,
                [
                    cdk.CfnParameter(
                        self,
                        "BucketName",
                        default="impulse-documents",
                        description="S3 bucket for document storage",
                    ).value_as_string,
                ],
            ),
            versioned=False,
            removal_policy=cdk.RemovalPolicy.RETAIN,
            auto_delete_objects=False,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            cors=[
                s3.CorsRule(
                    allowed_methods=[
                        s3.HttpMethods.GET,
                        s3.HttpMethods.PUT,
                    ],
                    allowed_origins=["*"],  # Tighten in production
                    allowed_headers=["*"],
                    max_age=3600,
                )
            ],
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=cdk.Duration.days(90),
                        )
                    ],
                )
            ],
        )

        # ── ECR repositories for ECS task containers ─────────────────────

        self.image_processing_repo = ecr.Repository(
            self,
            "ImageProcessingRepo",
            repository_name="impulse/image-processing",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            image_scan_on_push=True,
        )

        self.extraction_repo = ecr.Repository(
            self,
            "ExtractionRepo",
            repository_name="impulse/document-extraction",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            image_scan_on_push=True,
        )

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(self, "BucketArn", value=self.bucket.bucket_arn)
        cdk.CfnOutput(
            self,
            "ImageRepoUri",
            value=self.image_processing_repo.repository_uri,
        )
        cdk.CfnOutput(
            self,
            "ExtractionRepoUri",
            value=self.extraction_repo.repository_uri,
        )
