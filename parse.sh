#!/bin/bash

echo "Model,BatchSize,nGPUs,platform,use_hvd,use_fp16,ips"

for f in `ls -d *`; do
    hvd=`echo $f | grep hvd | sed -e 's/.*hvd.*/hvd/g'`
    fp16=`echo $f | grep fp16 | sed -e 's/.*fp16.*/fp16/g'`
    msg=`echo $f | sed -e 's/run_//g' | sed -e 's/_[0-9]*-[0-9]*.log//g' | sed -e 's/_fp16.*//g' | sed -e 's/_hvd.*//g' | sed -e 's/_/,/g' | sed -e 's/gbs//g' | sed -e 's/GPUs//g'`
    msg+=",${hvd},${fp16}"
    scores=`grep -inr "img\/sec" $f | tail -n 1 | sed -e 's/.*:\s*//g'`
    msg+=",${scores}"
    echo $msg
done
