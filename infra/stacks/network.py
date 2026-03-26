"""Network stack: VPC with private subnets and VPC endpoints."""

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2


class NetworkStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ── VPC ──────────────────────────────────────────────────────────

        self.vpc = ec2.Vpc(
            self,
            "ImpulseVpc",
            max_azs=2,
            nat_gateways=1,  # Needed for Fargate tasks to reach internet
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # ── VPC Endpoints (reduce NAT costs for high-volume S3 traffic) ──

        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        self.vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
        )

        self.vpc.add_interface_endpoint(
            "EcrEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR,
        )

        self.vpc.add_interface_endpoint(
            "EcrDockerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
        )

        self.vpc.add_interface_endpoint(
            "CloudWatchLogsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
        )

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
