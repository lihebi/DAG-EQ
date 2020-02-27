#!/bin/bash

if [ ! -d back ]; then
    mkdir back
fi

# TODO add archive message
tar zcvf back/archive-$(date +%m.%d.%y-%H.%M.%S).tar.gz tensorboard_logs/
