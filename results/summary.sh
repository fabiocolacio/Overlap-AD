#!/usr/bin/env bash

for file in *.txt; do 
    echo "Results from $file:"
    tail -n4 "$file"
    printf "\n"
done

