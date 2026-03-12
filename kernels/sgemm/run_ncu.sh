ncu -k regex:"gemm|Kernel" -c 4 --set full -o profile_res -f python test_for_ncu.py
