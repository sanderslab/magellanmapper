# Interface with AWS
# Author: David Young, 2018
"""Connect Clrbrain with AWS such as S3 and EC2.

Attributes:
"""

import boto3
from botocore.exceptions import ClientError

from clrbrain import cli
from clrbrain import config

def start_instances(tag_name, ami_id, instance_type, subnet_id, sec_group, ebs):
    client = boto3.client("ec2")
    #print(ec2.describe_instances())
    
    mappings = []
    for i in range(len(ebs)):
        device = ebs[i]
        name = "/dev/xvda"
        if i > 0:
            # iterate alphabetically starting with f since i >= 1
            name = "/dev/sd{}".format(chr(ord("e") + i))
        mapping = {
            "DeviceName": name, 
            "Ebs": {
                "VolumeSize": device
            }
        }
        mappings.append(mapping)
    
    res = boto3.resource("ec2")
    try:
        result = res.create_instances(
            MinCount=1, MaxCount=1, 
            ImageId=ami_id, InstanceType=instance_type, 
            NetworkInterfaces=[{
                "DeviceIndex": 0, 
                "AssociatePublicIpAddress": True, 
                "SubnetId": subnet_id, 
                "Groups": [sec_group]
            }], 
            BlockDeviceMappings=mappings, 
            TagSpecifications=[{
                "ResourceType": "instance", 
                "Tags": [{
                    "Key": "Name", 
                    "Value": tag_name
                }]
            }], 
            DryRun=False)
        print(result)
    except ClientError as e:
        print(e)

def stop_instances():
    res = boto3.resource("ec2")

if __name__ == "__main__":
    cli.main(True)
    start_instances(*config.ec2_start)
