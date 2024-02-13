### For getting clock frequency   
```bash
$ nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits
```

### Query the VBIOS version of each device

```bash
$ nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv
```

### Continuously provide time stamped power and clock

```bash
$ nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1
```