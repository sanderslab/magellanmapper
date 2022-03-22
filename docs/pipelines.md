# Running Pipelines in MagellanMapper

TODO: update for new pipelines

## Start a processing pipeline

Automated processing will attempt to scale based on your system resources but may require some manual intervention. This pipeline has been tested on a Macbook Pro laptop and AWS EC2 Linux (RHEL and Amazon Linux based) instances.

Optional dependencies:

- ImageJ/Fiji with the BigStitcher plugin: required for tile stitching; downloaded automatically onto a server when running `deploy.sh`
- ImageMagick: required for exporting a stack of planes to an animated GIF file
- FFMpeg: required to export a stack to a movie format such as MP4
- [Slack incoming webhook](https://api.slack.com/incoming-webhooks): to notify when tile stitching alignment is ready for verification and pipeline has completed

### Local
Run a pipeline in `pipelines.sh`.

For example, load a `.czi` file and display in the GUI, which will import the file into a Numpy format for faster future loading:

```
bin/pipelines.sh -i data/HugeImage.czi
```

To sitch a multi-tile image and perform cell detection on the entire image, which will load BigStitcher in ImageJ/Fiji for tile stitching:

```
bin/pipelines.sh -i data/HugeImage.czi -p full
```

See `bin/pipelines.sh` for additional sample commands for common scenarios, such as cell detection on a small region of interest. The file can be edited directly to load the same image, for example.

### Server

You can launch a standard server, deploy MagellanMapper code, and run a pipeline. See [tools for AWS cloud management](cloud_aws.sh) for more details. 
