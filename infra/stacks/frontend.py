"""Frontend stack: S3 bucket + CloudFront distribution for static hosting."""

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_cloudfront as cloudfront
import aws_cdk.aws_cloudfront_origins as origins
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_deployment as s3deploy


class FrontendStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ── S3 bucket for static site ────────────────────────────────────

        site_bucket = s3.Bucket(
            self,
            "ImpulseFrontendBucket",
            bucket_name=f"impulse-frontend-{cdk.Aws.ACCOUNT_ID}",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ── CloudFront distribution ──────────────────────────────────────

        distribution = cloudfront.Distribution(
            self,
            "ImpulseFrontendDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            default_root_object="index.html",
            # Handle SPA client-side routing: return index.html for all 404s
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=cdk.Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=cdk.Duration.seconds(0),
                ),
            ],
        )

        # ── Deploy static files ──────────────────────────────────────────

        s3deploy.BucketDeployment(
            self,
            "DeployFrontend",
            sources=[s3deploy.Source.asset("../frontend/out")],
            destination_bucket=site_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(
            self,
            "CloudFrontUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="Impulse frontend URL",
        )

        cdk.CfnOutput(
            self,
            "DistributionId",
            value=distribution.distribution_id,
        )

        cdk.CfnOutput(
            self,
            "FrontendBucketName",
            value=site_bucket.bucket_name,
        )
