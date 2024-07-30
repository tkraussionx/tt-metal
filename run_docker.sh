##!/bin/bash

 docker run -it --rm \
 --name my_docker4 \
 --env P_TIMEOUT=86400 \
 --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env LANG=en_US.UTF-8 --env LANGUAGE=en_US:en --env LC_ALL=en_US.UTF-8 \
 --volume /dev/hugepages:/dev/hugepages --volume /dev/hugepages-1G:/dev/hugepages-1G --volume /proj_sw:/proj_sw --volume /home/software/syseng:/home/software/syseng --volume /home/ppopovic:/home/ppopovic --volume /etc/udev/rules.d:/etc/udev/rules.d --volume /mnt/motor:/mnt/motor --volume /lib/modules:/lib/modules --volume /localdev/ppopovic:/localdev/ppopovic --volume /home_mnt:/home_mnt --volume /var/run/docker.sock:/var/run/docker.sock --volume /var/run/tenstorrent:/var/run/tenstorrent \
 --device=/dev/tenstorrent/0 \
 --device=/dev/tenstorrent/1 \
 --device=/dev/tenstorrent/2 \
 --device=/dev/tenstorrent/3 \
 ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:v0.51.0-rc13-dev \
 /bin/bash
