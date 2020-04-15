# Interface with AWS
# Author: David Young, 2018, 2019
"""Connect MagellanMapper with AWS such as S3 and EC2.

Attributes:
"""

import os
import multiprocessing as mp
from pprint import pprint

import boto3
import boto3.session
import botocore
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
from botocore.exceptions import ClientError

from magmap.io import cli
from magmap.io import df_io
from magmap.io import libmag
from magmap.settings import config

_EC2_STATES = (
    "pending", "running", "shutting-down", "terminated", "stopping", "stopped")


def instance_info(instance_id, get_ip):
    """Show settings for a given instance.
    
    Args:
        instance_id: ID of instance to query.
        get_ip: True to get instance IP, which will require waiting if the 
            instance is not yet running.
    
    Returns:
        Tuple of ``(instance_id, instance_ip)``.
    """
    # run in separate session since each resource shares data
    session = boto3.session.Session()
    ec2 = session.resource("ec2")
    instance = ec2.Instance(instance_id)
    image_id = instance.image_id
    tags = instance.tags
    instance_ip = "n/a"
    if get_ip:
        print("checking instance {}".format(instance))
        instance.wait_until_running()
        instance.load()
        instance_ip = instance.public_ip_address
    # show tag info but not saving for now since not currently used
    print("instance ID: {}, image ID: {}, tags: {}, IP: {}"
          .format(instance_id, image_id, tags, instance_ip))
    return instance_id, instance_ip


def show_instances(instances, get_ip=False):
    """Show settings for instances.
    
    Args:
        instances: List of instance objects to query.
        get_ip: True to get instance IP; defaults to False.
    
    Returns:
        Dictionary of ``instance_id: instance_ip`` entries.
    """
    # show instance info in multiprocessing to allow waiting for 
    # each instance to start running
    pool = mp.Pool()
    pool_results = []
    for instance in instances:
        pool_results.append(
            pool.apply_async(instance_info, args=(instance.id, get_ip)))
    info = {}
    for result in pool_results:
        inst_id, inst_ip = result.get()
        info[inst_id] = inst_ip
    pool.close()
    pool.join()
    return info


def start_instances(tag_name, ami_id, instance_type, subnet_id, sec_group, 
                    key_name, ebs, max_count=1, snapshot_ids=None):
    """Start EC2 instances.
    
    Args:
        tag_name: Name of tag.
        ami_id: ID for AMI from which to create instances.
        instance_type: EC2 instance type, such as "t2.micro".
        subnet_id: ID for subnet.
        sec_group: ID for security group.
        key_name: Name of key-pair to access EC2 instance with PEM.
        ebs: List of sizes in GB for each EBS instance to attach. Can also 
            be a single value, which will be converted to a tuple.
        max_count: Target number of instances to create; defaults to 1.
        snapshot_ids: List of snapshot IDs; defaults to None. Any empty 
            values will be ignored. Can also be a single value, which will be 
            converted to a tuple.
    """
    # convert single value ebs and snapshots into tuples
    if not isinstance(ebs, (tuple, list)): ebs = (ebs, )
    if snapshot_ids and not isinstance(snapshot_ids, (tuple, list)):
        snapshot_i = tuple(snapshot_ids, )
    
    mappings = []
    for i in range(len(ebs)):
        # parse EBS device block mappings
        device = ebs[i]
        name = "/dev/sda1" # default root vol at least on m5 series
        if i > 0:
            # iterate alphabetically starting with f since i >= 1
            name = "/dev/sd{}".format(chr(ord("e") + i))
        # use gp2 since otherwise may default to "standard" (magnetic HDD)
        ebs_dict = {
            "VolumeSize": device, 
            "VolumeType": "gp2"
        }
        if snapshot_ids and snapshot_ids[i]: 
            ebs_dict["SnapshotId"] = snapshot_ids[i]
        mapping = {
            "DeviceName": name, 
            "Ebs": ebs_dict
        }
        mappings.append(mapping)
    
    res = boto3.resource("ec2")
    try:
        instances = res.create_instances(
            MinCount=1, MaxCount=max_count, 
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
        show_instances(instances, True)
        
    except ClientError as e:
        print(e)


def terminate_instances(instance_ids):
    """Terminate instances with the given IDs.
    
    Args:
        instance_ids: List of IDs of instances to terminate.
    """
    client = boto3.client("ec2")
    try:
        result = client.terminate_instances(
            InstanceIds=instance_ids, DryRun=False)
        pprint(result)
    
    except ClientError as e:
        print(e)


def split_s3_path(path):
    """Split an S3 path into its bucket and key components.

    Args:
        path (str): S3 path, assumed to begin with ``s3://``.

    Returns:
        str, str: Bucket name and key.

    """
    path_split = path.split("/")
    bucket_name = None
    key = None
    num_path_split = len(path_split)
    if num_path_split > 2:
        bucket_name = path_split[2]
    if num_path_split > 3:
        key = "/".join(path_split[3:])
    return bucket_name, key


def list_instances(state=None, image_id=None):
    """List instances with the given parameters.
    
    Filters that are None will be ignored.
    
    Args:
        state: Filter instances by this state; defaults to None.
        image_id: Filter instances by this image ID; defaults to None.
    """
    res = boto3.resource("ec2")
    try:
        # filter instances by state
        filters = []
        names = ("instance-state-name", "image-id")
        vals = (state, image_id)
        for name, val in zip(names, vals):
            if val is not None:
                val = val if isinstance(val, (tuple, list)) else [val]
                filters.append({
                    "Name": name, 
                    "Values": val
                })
        instances = res.instances.filter(Filters=filters)
        print("listing instances with state {}:".format(state))
        info = show_instances(instances, get_ip=state==_EC2_STATES[1])
        
        # show IDs and IPs as contiguous lists for faster access
        print("\nall instance IDs ({}):".format(len(info.keys())))
        for key in info.keys():
            print(key)
        print("\nall instance IPs:")
        for val in info.values():
            print(val)
        
    except ClientError as e:
        print(e)


def get_bucket_size(name, keys=None, suffix=None, versions=False):
    """Get the tot_size of an AWS S3 bucket.

    Args:
        name (str): Name of bucket.
        keys (List[str]): Sequence of keys within the bucket to include
            sizes of only these files; defaults to None.
        suffix (str): String to append to output CSV file; defaults to None.
        versions (bool): True to get all object versions, including
            deleted objects; False to get only the current versions; defaults
            to False.

    Returns:
        float, :obj:`pd.DataFrame`, :obj:`pd.DataFrame`: Size of bucket in
        bytes; a dataframe of keys and associated sizes; and a dataframe
        of missing keys from ``keys``, or None if ``keys`` is not given.

    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(name)
    tot_size = 0
    obj_sizes = {}
    objs = bucket.object_versions.all() if versions else bucket.objects.all()
    for obj in objs:
        if not keys or obj.key in keys:
            obj_sizes.setdefault("Bucket", []).append(bucket.name)
            obj_sizes.setdefault("Key", []).append(obj.key)
            size = obj.size
            obj_sizes.setdefault("Size", []).append(size)
            if size:
                # skip delete markers, which have a size of None
                tot_size += obj.size
            if versions:
                obj_sizes.setdefault("Version_id", []).append(obj.version_id)
                obj_sizes.setdefault("Last_modified", []).append(obj.last_modified)

    out_path = "bucket_{}".format(bucket.name)
    if suffix:
        out_path = libmag.insert_before_ext(out_path, suffix, "_")
    df_missing = None
    if keys:
        keys_missing = []
        obj_keys = obj_sizes.keys()
        for key in keys:
            if key not in obj_keys:
                keys_missing.append(key)
        # print("Missing keys:\n", "\n".join(keys_missing))
        df_missing = df_io.dict_to_data_frame(
            {"Keys_missing": keys_missing},
            libmag.insert_before_ext(out_path, "_missing"))

    df = df_io.dict_to_data_frame(obj_sizes, out_path)
    print("{} bucket total tot_size (GiB): {}"
          .format(bucket.name, libmag.convert_bin_magnitude(tot_size, 3)))
    return tot_size, df, df_missing


def load_s3_file(bucket_name, key):
    """Load a file in AWS S3, retrieving the metadata without the object itself.

    Args:
        bucket_name (str): Name of bucket.
        key (str): Key within bucket.

    Returns:
        :obj:`boto3.resources.factory.s3.Object`: The S3 object if loaded
        successfully, otherwise None.

    """
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, key)
    try:
        obj.load()
        return obj
    except ClientError as e:
        print(e)
        print("Unable to load object at bucket \"{}\", key \"{}\""
              .format(bucket_name, key))
    return None


def download_s3_file(bucket_name, key, out_path=None, dryrun=False):
    """Download a file in AWS S3.

    Args:
        bucket_name (str): Name of bucket.
        key (str): Key within bucket.
        out_path (str): Output path; defaults to None to use the basename
            of ``key``.
        dryrun (bool): True to print paths without downloading;
            defaults to False.

    Returns:
        bool: True if the file was successfully downloaded, False if otherwise.

    """
    if out_path is None:
        out_path = os.path.basename(key)
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, key)

    print("Downloading bucket \"{}\", key \"{}\" to \"{}\""
          .format(bucket_name, key, out_path))
    with open(out_path, "wb") as f:
        if dryrun:
            # skip downloading while still testing local file write access
            print("Download of \"{}\" set to dry run, so skipping".format(key))
            return True
        # download as a managed transfer with multipart download;
        # flush as suggested in this issue:
        # https://github.com/boto/boto3/issues/1304
        try:
            obj.download_fileobj(f)
            f.flush()
            return True
        except (RetriesExceededError, ClientError) as e:
            print(e)
    return False


def upload_s3_file(path, bucket_name, key, dryrun=False):
    """Upload a file to AWS S3.

    Args:
        path (str): Path of file to upload.
        bucket_name (str): Name of bucket.
        key (str): Destination key in bucket for upload.
        dryrun (bool): True to print paths without uploading; defaults to False.

    Returns:
        bool: True if the file was successfully uploaded, False if otherwise.

    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    with open(path, "rb") as f:
        print("Uploading file \"{}\" to bucket \"{}\", key \"{}\""
              .format(path, bucket_name, key))
        if dryrun:
            # skip uploading while still testing local file access
            print("Upload of \"{}\" set to dry run, so skipping".format(path))
            return True
        # upload as a managed transfer with multipart download
        try:
            bucket.upload_fileobj(f, key)
            return True
        except S3UploadFailedError as e:
            print(e)
    return False


def delete_s3_file(bucket_name, key, hard=False, dryrun=False):
    """Delete a file on AWS S3.

    Args:
        bucket_name (str): Name of bucket.
        key (str): Key within bucket.
        hard (bool): True to delete all versions associated with the key
            including any delete markers, effectively deleting the
            object on S3 permanently. Defaults to False, in which case
            only a delete marker will be applied if versioning is on;
            without versioning, the file will be permanently deleted.
        dryrun (bool): True to print paths without uploading; defaults to False.

    Returns:
        bool: True if the file was successfully deleted, False if otherwise.
        For versioned files, all versions of the given object must have
        been deleted without error to return True.

    """
    s3 = boto3.resource("s3")
    print("Deleting file object at bucket \"{}\", key \"{}\" (hard delete {})"
          .format(bucket_name, key, hard))
    if hard:
        bucket = s3.Bucket(bucket_name)
        vers = bucket.object_versions.filter(Prefix=key)
        for ver in vers:
            if ver.object_key == key:
                print("Permanently deleting \"{}\", versionId \"{}\""
                      .format(ver.object_key, ver.id))
                if dryrun:
                    print("Deletion of \"{}\" set to dry run, so skipping"
                          .format(ver.object_key))
                else:
                    ver.delete()
        return True
    else:
        obj = s3.Object(bucket_name, key)
        try:
            print("Deleting (or setting delete marker for) \"{}\", versionId \"{}\""
                  .format(obj.key, obj.version_id))
            if dryrun:
                print("Deletion of {} set to dry run, so skipping"
                      .format(obj.key))
            else:
                obj.delete()
            return True
        except botocore.exceptions.ClientError as e:
            print(e)
            print("Could not find key \"{}\" to delete".format(key))
    return False


if __name__ == "__main__":
    cli.main(True)
    if config.ec2_start:
        start_instances(*config.ec2_start[:-1], **config.ec2_start[-1])
    if config.ec2_list:
        if len(config.ec2_list) == 1:
            # no parameters required, so may have list of only parameter dict
            list_instances(**config.ec2_list[0])
        else:
            list_instances(*config.ec2_list[:-1], **config.ec2_list[-1])
    if config.ec2_terminate:
        terminate_instances(config.ec2_terminate)
