#!/bin/bash

import javabridge as jb
import bioformats as bf

filename = 'P21_L5_CONT_DENDRITE.czi'
jars = bf.JARS # + [path]
#jb.start_vm(class_path=jars, max_heap_size='2G')
jb.start_vm(class_path=bf.JARS)

metadata = bf.OMEXML(bf.get_omexml_metadata(filename))
print(metadata.image_count)
print(metadata.image().AcquisitionDate)

jb.kill_vm()