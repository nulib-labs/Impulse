"""API stack: API Gateway + Lambda handlers for job management."""

from pathlib import Path

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_apigateway as apigw
import aws_cdk.aws_cognito as cognito
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_stepfunctions as sfn

_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent / "src")


class ApiStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        user_pool: cognito.IUserPool,
        bucket: s3.IBucket,
        state_machine: sfn.IStateMachine,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # ── Shared Lambda environment ────────────────────────────────────

        common_env = {
            "IMPULSE_BUCKET": bucket.bucket_name,
            "MONGODB_SECRET_ID": "impulse/mongodb-uri",
            "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            "USER_POOL_ID": user_pool.user_pool_id,
        }

        common_policy = iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
            ],
            resources=[
                bucket.bucket_arn,
                f"{bucket.bucket_arn}/*",
            ],
        )

        secrets_policy = iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:secret:impulse/*",
            ],
        )

        sfn_policy = iam.PolicyStatement(
            actions=[
                "states:StartExecution",
                "states:DescribeExecution",
                "states:GetExecutionHistory",
                "states:ListMapRuns",
                "states:DescribeMapRun",
                "states:ListExecutions",
            ],
            resources=[
                state_machine.state_machine_arn,
                f"{state_machine.state_machine_arn}/*",
                f"arn:aws:states:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:mapRun:*",
            ],
        )

        cognito_admin_policy = iam.PolicyStatement(
            actions=[
                "cognito-idp:AdminCreateUser",
                "cognito-idp:AdminDeleteUser",
                "cognito-idp:AdminDisableUser",
                "cognito-idp:AdminEnableUser",
                "cognito-idp:ListUsers",
                "cognito-idp:AdminGetUser",
            ],
            resources=[user_pool.user_pool_arn],
        )

        # ── Lambda: API handler ──────────────────────────────────────────
        # Bundle the source code with its lightweight dependencies
        # (loguru, pymongo, certifi).  boto3 is provided by the runtime.

        api_lambda = _lambda.Function(
            self,
            "ApiHandler",
            function_name="impulse-api",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="impulse.api.router.handler",
            code=_lambda.Code.from_asset(
                _SRC_DIR,
                bundling=cdk.BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_13.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install -r requirements-lambda.txt -t /asset-output"
                        " && cp -au . /asset-output"
                        " && rm -f /asset-output/requirements-lambda.txt",
                    ],
                ),
            ),
            memory_size=512,
            timeout=cdk.Duration.seconds(30),
            environment=common_env,
        )
        api_lambda.add_to_role_policy(common_policy)
        api_lambda.add_to_role_policy(secrets_policy)
        api_lambda.add_to_role_policy(sfn_policy)
        api_lambda.add_to_role_policy(cognito_admin_policy)
        bucket.grant_read_write(api_lambda)
        state_machine.grant_start_execution(api_lambda)

        # ── API Gateway ──────────────────────────────────────────────────

        api = apigw.RestApi(
            self,
            "ImpulseApi",
            rest_api_name="Impulse API",
            description="Impulse document processing API",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=[
                    "Content-Type",
                    "Authorization",
                    "X-Amz-Date",
                    "X-Api-Key",
                ],
            ),
        )

        # ── Cognito Authorizer ───────────────────────────────────────────

        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self,
            "ImpulseAuthorizer",
            cognito_user_pools=[user_pool],
            authorizer_name="impulse-cognito-authorizer",
        )

        method_opts = apigw.MethodOptions(
            authorizer=authorizer,
            authorization_type=apigw.AuthorizationType.COGNITO,
        )

        # ── Lambda integration ───────────────────────────────────────────

        lambda_integration = apigw.LambdaIntegration(api_lambda)

        # ── Proxy route ──────────────────────────────────────────────────
        # Use a single {proxy+} catch-all to avoid hitting the Lambda
        # resource policy size limit (20KB) with too many individual routes.
        # The Lambda router dispatches based on httpMethod + resource path.

        proxy = api.root.add_proxy(
            default_integration=lambda_integration,
            any_method=True,
            default_method_options=apigw.MethodOptions(
                authorization_type=apigw.AuthorizationType.COGNITO,
                authorizer=authorizer,
            ),
        )

        # ── Gateway Responses (CORS on auth failures) ────────────────────

        for response_type, name in [
            (apigw.ResponseType.UNAUTHORIZED, "GatewayUnauthorized"),
            (apigw.ResponseType.ACCESS_DENIED, "GatewayAccessDenied"),
            (apigw.ResponseType.DEFAULT_4_XX, "GatewayDefault4xx"),
            (apigw.ResponseType.DEFAULT_5_XX, "GatewayDefault5xx"),
        ]:
            api.add_gateway_response(
                name,
                type=response_type,
                response_headers={
                    "Access-Control-Allow-Origin": "'*'",
                    "Access-Control-Allow-Headers": "'Content-Type,Authorization,X-Amz-Date,X-Api-Key'",
                },
            )

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(
            self,
            "ApiUrl",
            value=api.url,
        )
