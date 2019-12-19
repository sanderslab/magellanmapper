# Notifications in MagellanMapper
# Author: David Young, 2018
"""Post MagellanMapper notifications.

Attributes:
"""

import json
from urllib import request

from magmap.io import cli
from magmap.settings import config
from magmap.io import libmag


def post(url, msg, attachment):
    post_fields = {"text": msg}
    if attachment:
        lines = libmag.last_lines(attachment, 20)
        if lines:
            attach_msg = "\n".join(lines)
            #print("got lines: {}".format(attach_msg))
            post_fields["attachments"] = [{"text": attach_msg}]
    header = {"Content-type": "application/json"}
    req = request.Request(url, json.dumps(post_fields).encode("utf8"), header)
    response = request.urlopen(req)
    print(response.read().decode("utf8"))
    return response

if __name__ == "__main__":
    print("Starting MagellanMapper notifier...")
    cli.main(True)
    post(config.notify_url, config.notify_msg, config.notify_attach)
