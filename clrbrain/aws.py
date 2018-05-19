# Interface with AWS
# Author: David Young, 2018
"""Connect Clrbrain with AWS such as S3 and EC2.

Attributes:
"""

import multiprocessing as mp

import boto3
import boto3.session
from botocore.exceptions import ClientError

from clrbrain import cli
from clrbrain import config

def instance_info(instance_id):
    # run in separate sessions since each resource shares data
    session = boto3.session.Session()
    ec2 = session.resource("ec2")
    instance = ec2.Instance(instance_id)
    print("checking instance {}".format(instance))
    instance.wait_until_running()
    instance.load()
    instance_id = instance.instance_id
    instance_ip = instance.public_ip_address
    print("instance ID: {}, IP: {}".format(instance_id, instance_ip))
    return instance_id, instance_ip

def start_instances(tag_name, ami_id, instance_type, subnet_id, sec_group, 
                    key_name, ebs):
    mappings = []
    for i in range(len(ebs)):
        device = ebs[i]
        name = "/dev/xvda"
        if i > 0:
            # iterate alphabetically starting with f since i >= 1
            name = "/dev/sd{}".format(chr(ord("e") + i))
        # use gp2 since otherwise may default to "standard" (magnetic HDD)
        mapping = {
            "DeviceName": name, 
            "Ebs": {
                "VolumeSize": device, 
                "VolumeType": "gp2"
            }
        }
        mappings.append(mapping)
    
    res = boto3.resource("ec2")
    try:
        instances = res.create_instances(
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
            KeyName=key_name, 
            DryRun=False)
        print(instances)
        
        # show instance info once running, multiprocessing to allow waiting for 
        # each instance to start running
        pool = mp.Pool()
        pool_results = []
        for instance in instances:
            pool_results.append(
                pool.apply_async(instance_info, args=(instance.id, )))
        for result in pool_results:
            inst_id, inst_ip = result.get()
        pool.close()
        pool.join()
            
    except ClientError as e:
        print(e)

def stop_instances():
    res = boto3.resource("ec2")

if __name__ == "__main__":
    cli.main(True)
    start_instances(*config.ec2_start)
