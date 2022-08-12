#!/bin/sh
ls | awk '{ printf "mv %s %03d.jpg\n", $0, NR }' | sh
