# locality_aware_nms_east
CUDA Implementation of Locality-aware NMS for EAST OCR Detection.

The efficient GPU NMS is firtly operated to obtain the IOUs. Then, based on the IOUs, the BBoxes are classified into different groups. Finally, the BBoxes in each group are merged as the non-overlapped one. The merging processes are executed parallelly by using the CUDA threads.

By using below command to generate the lib:
```
make all
```

The python API is in `nms_wrapper.py`.
