#!/bin/bash

# TODO add archive message
tar zcvf back/archive-$(date +%m.%d.%y-%H.%M.%S).tar.gz tensorboard_logs/
