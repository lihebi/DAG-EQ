#!/bin/bash

for name in *.hdf5; do
    mv $name ${name%.hdf5}_mat=COR.hdf5
done
            
